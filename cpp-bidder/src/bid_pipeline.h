#ifndef CPP_BIDDER_BID_PIPELINE_H
#define CPP_BIDDER_BID_PIPELINE_H

#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include "config.h"
#include "types.h"
#include "feature_store.h"
#include "model_session.h"
#include "micro_batcher.h"
#include "dedup_cache.h"
#include "circuit_breaker.h"
#include "fallback.h"
#include "metrics.h"

namespace bidder {

// The bid decision pipeline, decoupled from gRPC and written async-first so the
// async server never blocks a thread waiting for inference. process() runs the
// fast steps (dedup, circuit breaker, encode) inline, then either finishes
// immediately (cache hit / breaker open) or hands off to the micro-batcher and
// finishes from the batcher's callback when inference completes.
class BidPipeline {
 public:
  // status is one of: "ok", "cache", "fallback_open", "fallback_error",
  // "resource_exhausted". For "resource_exhausted" the decision is empty and the
  // server should return RESOURCE_EXHAUSTED; otherwise it returns the decision.
  using Done = std::function<void(const BidDecision& d, bool used_fallback,
                                  const char* status)>;

  BidPipeline(const ServerConfig& cfg, Metrics* metrics);

  // Run one request. done() fires exactly once, possibly on a batcher thread.
  void process(AdRequest req, Done done);

  // record an auction outcome (scoreboard metrics / future retraining hook)
  void notify_outcome(bool won, float clearing_price, float profit);

  // hot-reload the model; false if the new model fails validation
  bool reload_model(const std::string& path);

  bool ready() const { return ready_.load(); }
  void set_ready(bool r) { ready_.store(r); }
  const std::string& dsp_id() const { return cfg_.dsp_id; }
  Metrics* metrics() { return metrics_; }

 private:
  ServerConfig cfg_;
  Metrics* metrics_;

  FeatureStore store_;
  CircuitBreaker breaker_;
  DedupCache cache_;
  Fallback fallback_;
  std::unique_ptr<ModelSession> model_;
  std::unique_ptr<MicroBatcher> batcher_;

  std::atomic<bool> ready_{false};
  std::atomic<double> model_version_{1.0};
};

}  // namespace bidder

#endif
