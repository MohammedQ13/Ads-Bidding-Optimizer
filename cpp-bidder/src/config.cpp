#include "config.h"

#include <iostream>

#include <yaml-cpp/yaml.h>

namespace bidder {

// small helper: read a key if present, else keep the default
template <typename T>
static T get_or(const YAML::Node& node, const std::string& key, T def) {
  if (node[key]) {
    return node[key].as<T>();
  }
  return def;
}

ServerConfig ServerConfig::load(const std::string& path) {
  ServerConfig c;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const std::exception& e) {
    std::cerr << "config: could not load " << path << " (" << e.what()
              << "), using defaults" << std::endl;
    return c;
  }

  c.grpc_address = get_or<std::string>(root, "grpc_address", c.grpc_address);
  c.metrics_port = get_or<int>(root, "metrics_port", c.metrics_port);
  c.model_path = get_or<std::string>(root, "model_path", c.model_path);
  c.feature_config_path =
      get_or<std::string>(root, "feature_config_path", c.feature_config_path);

  c.dsp_id = get_or<std::string>(root, "dsp_id", c.dsp_id);
  c.strategy_type = get_or<std::string>(root, "strategy_type", c.strategy_type);
  c.bid_multiplier = get_or<float>(root, "bid_multiplier", c.bid_multiplier);

  c.cq_threads = get_or<int>(root, "cq_threads", c.cq_threads);
  c.batch_workers = get_or<int>(root, "batch_workers", c.batch_workers);
  c.batch_max_size = get_or<int>(root, "batch_max_size", c.batch_max_size);
  c.batch_timeout_us = get_or<int>(root, "batch_timeout_us", c.batch_timeout_us);
  c.queue_max_size = get_or<int>(root, "queue_max_size", c.queue_max_size);

  c.breaker_failure_threshold =
      get_or<int>(root, "breaker_failure_threshold", c.breaker_failure_threshold);
  c.breaker_cooldown_ms =
      get_or<int>(root, "breaker_cooldown_ms", c.breaker_cooldown_ms);

  c.dedup_cache_size = get_or<int>(root, "dedup_cache_size", c.dedup_cache_size);
  c.dedup_cache_ttl_ms =
      get_or<int>(root, "dedup_cache_ttl_ms", c.dedup_cache_ttl_ms);

  c.fallback_bid = get_or<float>(root, "fallback_bid", c.fallback_bid);
  c.fallback_shading = get_or<float>(root, "fallback_shading", c.fallback_shading);

  c.watch_model = get_or<bool>(root, "watch_model", c.watch_model);
  c.watch_interval_ms =
      get_or<int>(root, "watch_interval_ms", c.watch_interval_ms);

  if (root["fallback_by_advertiser"]) {
    for (auto it = root["fallback_by_advertiser"].begin();
         it != root["fallback_by_advertiser"].end(); ++it) {
      c.fallback_by_advertiser[it->first.as<std::string>()] =
          it->second.as<float>();
    }
  }

  return c;
}

}  // namespace bidder
