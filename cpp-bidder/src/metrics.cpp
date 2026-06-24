#include "metrics.h"

namespace bidder {

using namespace prometheus;

Metrics::Metrics(const std::string& bind, const std::string& dsp_id) {
  exposer_ = std::make_unique<Exposer>(bind);
  registry_ = std::make_shared<Registry>();

  const std::map<std::string, std::string> labels = {{"dsp_id", dsp_id}};

  request_family_ = &BuildCounter()
                         .Name("bid_request_total")
                         .Help("Total bid requests by outcome status")
                         .Labels(labels)
                         .Register(*registry_);
  // pre-create the counters for the statuses the hot path uses, so observe_request
  // is just a hash lookup + atomic increment, no locked Family::Add per request
  for (const char* s : {"ok", "cache", "fallback_open", "fallback_error",
                        "resource_exhausted"}) {
    req_counters_[s] = &request_family_->Add({{"status", s}});
  }

  // latency buckets in seconds: 1,2,5,10,25,50 ms
  Histogram::BucketBoundaries lat_buckets = {0.001, 0.002, 0.005,
                                             0.010, 0.025, 0.050};
  request_latency_ = &BuildHistogram()
                          .Name("bid_request_latency_seconds")
                          .Help("End to end GetBid latency")
                          .Labels(labels)
                          .Register(*registry_)
                          .Add({}, lat_buckets);

  Histogram::BucketBoundaries inf_buckets = {0.0001, 0.0002, 0.0005,
                                             0.001, 0.002, 0.005};
  inference_latency_ = &BuildHistogram()
                            .Name("bid_inference_latency_seconds")
                            .Help("ONNX Run() latency per batch")
                            .Labels(labels)
                            .Register(*registry_)
                            .Add({}, inf_buckets);

  Histogram::BucketBoundaries batch_buckets = {1, 2, 4, 8, 16, 32, 64};
  batch_size_ = &BuildHistogram()
                     .Name("bid_batch_size")
                     .Help("Inference batch size")
                     .Labels(labels)
                     .Register(*registry_)
                     .Add({}, batch_buckets);

  cache_hit_ = &BuildCounter()
                    .Name("bid_cache_hit_total")
                    .Help("Dedup cache hits")
                    .Labels(labels)
                    .Register(*registry_)
                    .Add({});

  fallback_ = &BuildCounter()
                   .Name("bid_fallback_total")
                   .Help("Requests served by the fallback bid")
                   .Labels(labels)
                   .Register(*registry_)
                   .Add({});

  auction_won_ = &BuildCounter()
                      .Name("bid_auction_won_total")
                      .Help("Auctions won")
                      .Labels(labels)
                      .Register(*registry_)
                      .Add({});

  auction_lost_ = &BuildCounter()
                       .Name("bid_auction_lost_total")
                       .Help("Auctions lost")
                       .Labels(labels)
                       .Register(*registry_)
                       .Add({});

  profit_total_ = &BuildCounter()
                       .Name("bid_profit_total")
                       .Help("Cumulative realized profit, fen")
                       .Labels(labels)
                       .Register(*registry_)
                       .Add({});

  last_clearing_price_ = &BuildGauge()
                              .Name("bid_last_clearing_price")
                              .Help("Clearing price of the most recent auction")
                              .Labels(labels)
                              .Register(*registry_)
                              .Add({});

  cold_start_ = &BuildGauge()
                     .Name("bid_cold_start_lookup_total")
                     .Help("Unknown vocab lookups since startup")
                     .Labels(labels)
                     .Register(*registry_)
                     .Add({});

  breaker_state_ = &BuildGauge()
                        .Name("bid_circuit_breaker_state")
                        .Help("0 closed, 1 open, 2 half-open")
                        .Labels(labels)
                        .Register(*registry_)
                        .Add({});

  queue_depth_ = &BuildGauge()
                      .Name("bid_request_queue_depth")
                      .Help("Pending requests in the batch queue")
                      .Labels(labels)
                      .Register(*registry_)
                      .Add({});

  model_version_ = &BuildGauge()
                        .Name("bid_model_version")
                        .Help("Loaded model version (timestamp or counter)")
                        .Labels(labels)
                        .Register(*registry_)
                        .Add({});

  exposer_->RegisterCollectable(registry_);
}

void Metrics::observe_request(const std::string& status, double latency_seconds) {
  auto it = req_counters_.find(status);
  if (it != req_counters_.end()) {
    it->second->Increment();
  } else {
    request_family_->Add({{"status", status}}).Increment();
  }
  request_latency_->Observe(latency_seconds);
}

void Metrics::observe_inference(double seconds) {
  inference_latency_->Observe(seconds);
}

void Metrics::observe_batch(int size) {
  batch_size_->Observe(static_cast<double>(size));
}

void Metrics::inc_cache_hit() { cache_hit_->Increment(); }

void Metrics::inc_fallback() { fallback_->Increment(); }

void Metrics::observe_outcome(bool won, double profit, double clearing_price) {
  if (won) {
    auction_won_->Increment();
    if (profit > 0.0) {
      profit_total_->Increment(profit);
    }
  } else {
    auction_lost_->Increment();
  }
  last_clearing_price_->Set(clearing_price);
}

void Metrics::set_cold_start(long total) {
  cold_start_->Set(static_cast<double>(total));
}

void Metrics::set_breaker_state(int code) {
  breaker_state_->Set(static_cast<double>(code));
}

void Metrics::set_queue_depth(int depth) {
  queue_depth_->Set(static_cast<double>(depth));
}

void Metrics::set_model_version(double version) { model_version_->Set(version); }

}  // namespace bidder
