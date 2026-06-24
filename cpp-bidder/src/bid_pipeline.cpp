#include "bid_pipeline.h"

#include "bid_optimizer.h"

namespace bidder {

BidPipeline::BidPipeline(const ServerConfig& cfg, Metrics* metrics)
    : cfg_(cfg),
      metrics_(metrics),
      store_(cfg.feature_config_path),
      breaker_(cfg.breaker_failure_threshold, cfg.breaker_cooldown_ms),
      cache_(cfg.dedup_cache_size, cfg.dedup_cache_ttl_ms),
      fallback_(cfg.fallback_bid, cfg.fallback_shading,
                cfg.fallback_by_advertiser) {
  model_ = std::make_unique<ModelSession>(cfg.model_path);

  Metrics* m = metrics_;
  MicroBatcher::StatsHook hook = [m](int bs, double secs) {
    if (m) {
      m->observe_batch(bs);
      m->observe_inference(secs);
    }
  };
  batcher_ = std::make_unique<MicroBatcher>(*model_, cfg.batch_workers,
                                            cfg.batch_max_size,
                                            cfg.queue_max_size, hook);
  if (metrics_) {
    metrics_->set_model_version(model_version_.load());
  }
  ready_.store(true);
}

void BidPipeline::process(AdRequest req, Done done) {
  // dedup: a retried request id within the TTL reuses the same decision
  BidDecision cached;
  if (cache_.get(req.request_id, cached)) {
    if (metrics_) {
      metrics_->inc_cache_hit();
    }
    done(cached, false, "cache");
    return;
  }

  // circuit breaker: if inference has been failing, skip it and bid the fallback
  if (!breaker_.allow_request()) {
    BidDecision d = fallback_.bid_for(req, cfg_.bid_multiplier);
    if (metrics_) {
      metrics_->inc_fallback();
    }
    done(d, true, "fallback_open");
    return;
  }

  EncodedFeatures enc = store_.encode(req);

  // hand off to the batcher; the callback finishes the request when inference
  // completes (on a batcher worker thread). req is captured by value so it
  // stays alive for the fallback path.
  float mult = cfg_.bid_multiplier;
  int num_bins = model_->num_bins();
  auto cb = [this, req, done, mult, num_bins](bool ok, std::vector<float>& probs) {
    if (!ok) {
      breaker_.record_failure();
      BidDecision d = fallback_.bid_for(req, mult);
      if (metrics_) {
        metrics_->inc_fallback();
      }
      done(d, true, "fallback_error");
      return;
    }
    breaker_.record_success();
    BidDecision d = optimize_bid(probs.data(), num_bins, req.impression_value, mult);
    cache_.put(req.request_id, d);
    if (metrics_) {
      metrics_->set_cold_start(store_.cold_start_count());
      metrics_->set_breaker_state(breaker_.state_code());
    }
    done(d, false, "ok");
  };

  if (!batcher_->submit(enc, std::move(cb))) {
    BidDecision empty;
    done(empty, false, "resource_exhausted");
  }
}

void BidPipeline::notify_outcome(bool won, float clearing_price, float profit) {
  if (metrics_) {
    metrics_->observe_outcome(won, profit, clearing_price);
  }
}

bool BidPipeline::reload_model(const std::string& path) {
  try {
    model_->reload(path);
  } catch (const std::exception& e) {
    return false;
  }
  double v = model_version_.load() + 1.0;
  model_version_.store(v);
  if (metrics_) {
    metrics_->set_model_version(v);
  }
  return true;
}

}  // namespace bidder
