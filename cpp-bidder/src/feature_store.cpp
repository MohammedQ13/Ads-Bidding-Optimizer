#include "feature_store.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace bidder {

using json = nlohmann::json;

// Day of week as Monday=0 .. Sunday=6, which is what pandas dt.weekday gives.
// The training code used that for the weekday sin/cos features, so we have to
// match it. Sakamoto's algorithm returns Sunday=0, so we shift it at the end.
static int weekday_monday0(int y, int m, int d) {
  static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
  if (m < 1 || m > 12 || d < 1) {
    return 0;
  }
  int yy = y;
  if (m < 3) {
    yy -= 1;
  }
  int sakamoto = (yy + yy / 4 - yy / 100 + yy / 400 + t[m - 1] + d) % 7;  // 0=Sun
  return (sakamoto + 6) % 7;  // shift so Monday=0
}

FeatureStore::FeatureStore(const std::string& config_path) {
  std::ifstream f(config_path);
  if (!f.is_open()) {
    throw std::runtime_error("cannot open feature config: " + config_path);
  }
  json cfg;
  f >> cfg;

  // categorical vocabs, kept in the model's feature order
  for (const auto& feat : cfg["cat_order"]) {
    std::string name = feat.get<std::string>();
    cat_order_.push_back(name);
    std::unordered_map<std::string, int> vocab;
    const auto& vj = cfg["cat_vocabs"][name];
    vocab.reserve(vj.size() * 2);
    for (auto it = vj.begin(); it != vj.end(); ++it) {
      vocab[it.key()] = it.value().get<int>();
    }
    cat_vocabs_.push_back(std::move(vocab));
  }

  // tag vocab (string token -> index, 0 is pad)
  const auto& tv = cfg["tag_vocab"];
  tag_vocab_.reserve(tv.size() * 2);
  for (auto it = tv.begin(); it != tv.end(); ++it) {
    tag_vocab_[it.key()] = it.value().get<int>();
  }

  mean_log_floor_ = cfg["cont_means"]["log_floor_price"].get<double>();
  std_log_floor_ = cfg["cont_stds"]["log_floor_price"].get<double>();
  mean_slot_area_ = cfg["cont_means"]["slot_area"].get<double>();
  std_slot_area_ = cfg["cont_stds"]["slot_area"].get<double>();
  mean_tag_count_ = cfg["cont_means"]["tag_count"].get<double>();
  std_tag_count_ = cfg["cont_stds"]["tag_count"].get<double>();

  tag_max_len_ = cfg.value("tag_max_len", 10);
  num_bins_ = cfg.value("num_bins", 200);
}

std::vector<std::string> FeatureStore::split_tags(const std::string& raw) {
  std::vector<std::string> out;
  if (raw.empty() || raw == "null" || raw == "NaN") {
    return out;
  }
  size_t start = 0;
  while (start <= raw.size()) {
    size_t comma = raw.find(',', start);
    if (comma == std::string::npos) {
      std::string tok = raw.substr(start);
      if (!tok.empty()) {
        out.push_back(tok);
      }
      break;
    }
    std::string tok = raw.substr(start, comma - start);
    if (!tok.empty()) {
      out.push_back(tok);
    }
    start = comma + 1;
  }
  return out;
}

EncodedFeatures FeatureStore::encode(const AdRequest& req) const {
  EncodedFeatures out;

  // Look up each category in the fixed model order. The numeric ones (region,
  // city, ...) are looked up by their number turned into a string; advertiser_id
  // and domain are already strings.
  std::string keys[9];
  keys[0] = std::to_string(req.region);
  keys[1] = std::to_string(req.city);
  keys[2] = req.domain;
  keys[3] = std::to_string(req.ad_exchange);
  keys[4] = std::to_string(req.slot_width);
  keys[5] = std::to_string(req.slot_height);
  keys[6] = std::to_string(req.slot_visibility);
  keys[7] = std::to_string(req.slot_format);
  keys[8] = req.advertiser_id;
  for (int i = 0; i < 9; i++) {
    const auto& vocab = cat_vocabs_[i];
    auto it = vocab.find(keys[i]);
    if (it == vocab.end()) {
      out.cat[i] = 0;
      cold_start_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
      out.cat[i] = it->second;
    }
  }

  // Continuous features. Only the first three get standardized (z-scored); the
  // rest are used as-is (the 0/1 flags and the sin/cos time features).
  double floor = req.slot_floor_price;
  if (floor < 0.0) {
    floor = 0.0;
  }
  double log_floor = std::log1p(floor);
  double slot_area = static_cast<double>(req.slot_width) * static_cast<double>(req.slot_height);

  // tag_count is the raw number of tokens, including ones not in the vocab
  int tag_count = 0;
  for (const auto& t : req.user_tags) {
    if (!t.empty()) {
      tag_count++;
    }
  }

  out.cont[0] = static_cast<float>((log_floor - mean_log_floor_) / std_log_floor_);
  out.cont[1] = static_cast<float>((slot_area - mean_slot_area_) / std_slot_area_);
  out.cont[2] = static_cast<float>((static_cast<double>(tag_count) - mean_tag_count_) / std_tag_count_);
  out.cont[3] = (req.slot_floor_price > 0.0f) ? 1.0f : 0.0f;  // has_floor_price

  // time features from the YYYYMMDDHHmmss timestamp
  int64_t ts = req.timestamp;
  int64_t date_part = ts / 1000000;
  int64_t time_part = ts % 1000000;
  int year = static_cast<int>(date_part / 10000);
  int month = static_cast<int>((date_part / 100) % 100);
  int day = static_cast<int>(date_part % 100);
  int hour = static_cast<int>(time_part / 10000);
  if (hour < 0 || hour > 23) {
    hour = 0;
  }
  int weekday = weekday_monday0(year, month, day);

  out.cont[4] = (weekday >= 5) ? 1.0f : 0.0f;  // is_weekend
  const double two_pi = 2.0 * M_PI;
  out.cont[5] = static_cast<float>(std::sin(two_pi * hour / 24.0));   // hour_sin
  out.cont[6] = static_cast<float>(std::cos(two_pi * hour / 24.0));   // hour_cos
  out.cont[7] = static_cast<float>(std::sin(two_pi * weekday / 7.0)); // weekday_sin
  out.cont[8] = static_cast<float>(std::cos(two_pi * weekday / 7.0)); // weekday_cos

  // tag ids: keep up to tag_max_len known tags, skip unknown ones, pad with 0
  for (int i = 0; i < 10; i++) {
    out.tags[i] = 0;
  }
  int j = 0;
  for (const auto& tok : req.user_tags) {
    if (j >= tag_max_len_) {
      break;
    }
    if (tok.empty()) {
      continue;
    }
    auto it = tag_vocab_.find(tok);
    if (it == tag_vocab_.end() || it->second == 0) {
      continue;  // tag we never saw in training, skip it
    }
    out.tags[j] = it->second;
    j++;
  }

  return out;
}

}  // namespace bidder
