#ifndef CPP_BIDDER_FEATURE_STORE_H
#define CPP_BIDDER_FEATURE_STORE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <cstdint>

#include "types.h"

namespace bidder {

// Loads feature_config.json (saved next to the ONNX model during training) and
// turns a raw ad request into the numbers the model expects. This has to match
// the Python training code exactly. If it doesn't, the model still runs but
// every bid is quietly wrong, so there's a parity test that checks it against
// numbers Python produced.
class FeatureStore {
 public:
  // Loads the config from a JSON file. Throws std::runtime_error on failure.
  explicit FeatureStore(const std::string& config_path);

  // Turn one request into model inputs. Categories or tags we never saw in
  // training map to index 0 (the "unknown" slot) and add to a cold-start count.
  EncodedFeatures encode(const AdRequest& req) const;

  int num_bins() const { return num_bins_; }

  // How many unknown lookups we've had since startup. The metrics layer reports
  // this as the cold-start counter.
  long cold_start_count() const { return cold_start_count_.load(); }

  // Split a comma-separated tag string the same way the training code did:
  // "null"/"NaN"/empty mean no tags, otherwise split on commas and keep the
  // non-empty pieces.
  static std::vector<std::string> split_tags(const std::string& raw);

 private:
  // one value->index map per categorical feature, keyed by the string form of
  // the value (json object keys are always strings)
  std::vector<std::unordered_map<std::string, int>> cat_vocabs_;
  std::vector<std::string> cat_order_;
  std::unordered_map<std::string, int> tag_vocab_;

  // standardization stats for the three continuous features that need it
  double mean_log_floor_ = 0.0, std_log_floor_ = 1.0;
  double mean_slot_area_ = 0.0, std_slot_area_ = 1.0;
  double mean_tag_count_ = 0.0, std_tag_count_ = 1.0;

  int tag_max_len_ = 10;
  int num_bins_ = 200;

  mutable std::atomic<long> cold_start_count_{0};
};

}  // namespace bidder

#endif
