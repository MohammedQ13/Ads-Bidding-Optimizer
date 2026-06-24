#ifndef CPP_BIDDER_CONFIG_H
#define CPP_BIDDER_CONFIG_H

#include <string>
#include <unordered_map>

namespace bidder {

// All the settings, loaded from a YAML file. Everything has a default, so even
// with no config file the server still starts and works. The same program runs
// as any "company" just by changing dsp_id, strategy_type, and bid_multiplier.
struct ServerConfig {
  // networking
  std::string grpc_address = "0.0.0.0:50051";
  int metrics_port = 9100;

  // model + features
  std::string model_path = "models/bid_model.onnx";
  std::string feature_config_path = "models/feature_config.json";

  // identity / strategy
  std::string dsp_id = "company-a";
  std::string strategy_type = "profit_max";
  float bid_multiplier = 1.0f;

  // micro-batcher + work queue
  int batch_workers = 2;
  int batch_max_size = 32;
  int batch_timeout_us = 1000;
  int queue_max_size = 4096;

  // number of completion-queue polling threads for the async server. A few is
  // plenty since they never block on inference (the batcher does that).
  int cq_threads = 4;

  // circuit breaker
  int breaker_failure_threshold = 5;
  int breaker_cooldown_ms = 5000;

  // dedup cache
  int dedup_cache_size = 10000;
  int dedup_cache_ttl_ms = 500;

  // fallback bid
  float fallback_bid = 80.0f;       // historical average clearing price, fen
  float fallback_shading = 0.7f;    // shade below the average to stay profitable
  std::unordered_map<std::string, float> fallback_by_advertiser;

  // hot-reload: poll the model file and swap it in when it changes (Phase 3)
  bool watch_model = false;
  int watch_interval_ms = 2000;

  // load from a YAML file; missing keys keep their defaults
  static ServerConfig load(const std::string& path);
};

}  // namespace bidder

#endif
