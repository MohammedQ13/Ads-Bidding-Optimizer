#ifndef CPP_BIDDER_METRICS_H
#define CPP_BIDDER_METRICS_H

#include <memory>
#include <string>
#include <unordered_map>

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

namespace bidder {

// The Prometheus metrics. This opens a /metrics page that Prometheus reads and
// Grafana graphs. Updating any of these is cheap (just atomic increments inside
// prometheus-cpp), so it's fine to do on every request.
class Metrics {
 public:
  // bind looks like "0.0.0.0:9100". dsp_id is attached as a label to every
  // metric, so one Prometheus can scrape all the companies and still tell them
  // apart.
  Metrics(const std::string& bind, const std::string& dsp_id);

  void observe_request(const std::string& status, double latency_seconds);
  void observe_inference(double seconds);
  void observe_batch(int size);
  void inc_cache_hit();
  void inc_fallback();
  // auction outcome reported back by the engine (for the strategy scoreboard)
  void observe_outcome(bool won, double profit, double clearing_price);
  void set_cold_start(long total);
  void set_breaker_state(int code);
  void set_queue_depth(int depth);
  void set_model_version(double version);

 private:
  std::unique_ptr<prometheus::Exposer> exposer_;
  std::shared_ptr<prometheus::Registry> registry_;

  prometheus::Family<prometheus::Counter>* request_family_;
  // pre-created per-status counters so the hot path skips the locked map lookup
  std::unordered_map<std::string, prometheus::Counter*> req_counters_;
  prometheus::Histogram* request_latency_;
  prometheus::Histogram* inference_latency_;
  prometheus::Histogram* batch_size_;
  prometheus::Counter* cache_hit_;
  prometheus::Counter* fallback_;
  prometheus::Counter* auction_won_;
  prometheus::Counter* auction_lost_;
  prometheus::Counter* profit_total_;
  prometheus::Gauge* last_clearing_price_;
  prometheus::Gauge* cold_start_;
  prometheus::Gauge* breaker_state_;
  prometheus::Gauge* queue_depth_;
  prometheus::Gauge* model_version_;
};

}  // namespace bidder

#endif
