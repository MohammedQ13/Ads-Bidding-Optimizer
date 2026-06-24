// Parity test: encode the same raw requests in C++ that generate_fixture.py
// encoded in Python, run the same ONNX model, compute the same bids, and check
// they match. If this passes, the C++ hot path reproduces the Python pipeline
// the reported regret numbers came from.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../src/feature_store.h"
#include "../src/model_session.h"
#include "../src/bid_optimizer.h"

using json = nlohmann::json;
using namespace bidder;

static int failures = 0;

static void check(bool ok, const std::string& msg) {
  if (!ok) {
    failures++;
    std::printf("  FAIL: %s\n", msg.c_str());
  }
}

int main(int argc, char** argv) {
  std::string fixture_path = "test/golden_fixture.json";
  std::string config_path = "models/feature_config.json";
  std::string model_path = "models/bid_model.onnx";
  if (argc > 1) fixture_path = argv[1];
  if (argc > 2) config_path = argv[2];
  if (argc > 3) model_path = argv[3];

  std::ifstream f(fixture_path);
  if (!f.is_open()) {
    std::printf("cannot open fixture: %s\n", fixture_path.c_str());
    return 2;
  }
  json fx;
  f >> fx;

  FeatureStore store(config_path);
  ModelSession model(model_path);
  int num_bins = fx["num_bins"].get<int>();

  check(store.num_bins() == num_bins, "num_bins mismatch (feature config)");
  check(model.num_bins() == num_bins, "num_bins mismatch (onnx model)");

  int case_idx = 0;
  for (const auto& c : fx["cases"]) {
    const auto& r = c["request"];
    std::string id = r.value("request_id", "case" + std::to_string(case_idx));

    AdRequest req;
    req.request_id = id;
    req.region = r["region"].get<int>();
    req.city = r["city"].get<int>();
    req.domain = r["domain"].get<std::string>();
    req.ad_exchange = r["ad_exchange"].get<int>();
    req.slot_width = r["slot_width"].get<int>();
    req.slot_height = r["slot_height"].get<int>();
    req.slot_visibility = r["slot_visibility"].get<int>();
    req.slot_format = r["slot_format"].get<int>();
    req.advertiser_id = r["advertiser_id"].get<std::string>();
    req.slot_floor_price = r["slot_floor_price"].get<float>();
    req.timestamp = r["timestamp"].get<int64_t>();
    req.user_tags = FeatureStore::split_tags(r["user_tags"].get<std::string>());

    EncodedFeatures enc = store.encode(req);

    // categorical indices must match exactly
    for (int i = 0; i < 9; i++) {
      int expect = c["cat"][i].get<int>();
      check(enc.cat[i] == expect,
            id + " cat[" + std::to_string(i) + "] got " +
                std::to_string(enc.cat[i]) + " want " + std::to_string(expect));
    }
    // tag indices must match exactly
    for (int i = 0; i < 10; i++) {
      int expect = c["tags"][i].get<int>();
      check(enc.tags[i] == expect,
            id + " tags[" + std::to_string(i) + "] got " +
                std::to_string(enc.tags[i]) + " want " + std::to_string(expect));
    }
    // continuous features within float tolerance
    for (int i = 0; i < 9; i++) {
      double expect = c["cont"][i].get<double>();
      double diff = std::fabs(static_cast<double>(enc.cont[i]) - expect);
      check(diff < 1e-4,
            id + " cont[" + std::to_string(i) + "] got " +
                std::to_string(enc.cont[i]) + " want " + std::to_string(expect) +
                " diff " + std::to_string(diff));
    }

    // run the model and compare probabilities
    std::vector<float> probs;
    model.run(enc.cat, enc.cont, enc.tags, 1, probs);
    check(static_cast<int>(probs.size()) == num_bins, id + " probs size");
    double max_prob_diff = 0.0;
    for (int i = 0; i < num_bins; i++) {
      double expect = c["probs"][i].get<double>();
      double diff = std::fabs(static_cast<double>(probs[i]) - expect);
      if (diff > max_prob_diff) max_prob_diff = diff;
    }
    check(max_prob_diff < 1e-4,
          id + " probs max diff " + std::to_string(max_prob_diff));

    // bids for each V value must match the Python grid optimizer exactly
    for (auto it = c["bids"].begin(); it != c["bids"].end(); ++it) {
      float V = std::stof(it.key());
      double expect_bid = it.value()["bid"].get<double>();
      BidDecision d = optimize_bid(probs.data(), num_bins, V, 1.0f);
      check(std::fabs(d.bid_price - expect_bid) < 1e-6,
            id + " bid V=" + it.key() + " got " + std::to_string(d.bid_price) +
                " want " + std::to_string(expect_bid));
    }

    case_idx++;
  }

  if (failures == 0) {
    std::printf("PARITY OK: %d cases, cold starts seen = %ld\n", case_idx,
                store.cold_start_count());
    return 0;
  }
  std::printf("PARITY FAILED: %d checks failed\n", failures);
  return 1;
}
