#include "async_server.h"

#include <chrono>
#include <cstring>

#include "feature_store.h"

namespace bidder {

using grpc::ServerAsyncResponseWriter;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;

// Base for every in-flight call. The CQ hands back the CallData pointer as the
// tag; Proceed advances its little state machine.
class CallData {
 public:
  virtual ~CallData() = default;
  virtual void Proceed(bool ok) = 0;
};

// Build the internal AdRequest from the protobuf request.
static AdRequest from_proto(const bidding::BidRequest& r) {
  AdRequest ad;
  ad.request_id = r.request_id();
  ad.region = r.region();
  ad.city = r.city();
  ad.domain = r.domain();
  ad.ad_exchange = r.ad_exchange();
  ad.slot_width = r.slot_width();
  ad.slot_height = r.slot_height();
  ad.slot_visibility = r.slot_visibility();
  ad.slot_format = r.slot_format();
  ad.advertiser_id = r.advertiser_id();
  ad.slot_floor_price = r.slot_floor_price();
  ad.impression_value = r.impression_value();
  ad.timestamp = r.timestamp();
  for (const auto& t : r.user_tags()) {
    auto parts = FeatureStore::split_tags(t);
    for (auto& p : parts) {
      ad.user_tags.push_back(p);
    }
  }
  return ad;
}

// ---- GetBid ----
class GetBidCall : public CallData {
 public:
  GetBidCall(AsyncBidServer* server, ServerCompletionQueue* cq)
      : server_(server), cq_(cq), responder_(&ctx_) {
    server_->service()->RequestGetBid(&ctx_, &req_, &responder_, cq_, cq_, this);
  }

  void Proceed(bool ok) override {
    if (state_ == PROCESS) {
      if (!ok) {  // server shutting down before this RPC arrived
        delete this;
        return;
      }
      new GetBidCall(server_, cq_);  // keep accepting new GetBids
      t0_ = std::chrono::steady_clock::now();
      AdRequest ad = from_proto(req_);
      server_->pipeline()->process(
          std::move(ad), [this](const BidDecision& d, bool fb, const char* st) {
            finish(d, fb, st);
          });
    } else {
      delete this;  // FINISH completed
    }
  }

 private:
  void finish(const BidDecision& d, bool used_fallback, const char* status) {
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0_).count();
    if (server_->pipeline()->metrics()) {
      server_->pipeline()->metrics()->observe_request(status, secs);
    }
    state_ = FINISH;
    if (std::strcmp(status, "resource_exhausted") == 0) {
      responder_.FinishWithError(
          Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "bid queue full"), this);
      return;
    }
    resp_.set_dsp_id(server_->pipeline()->dsp_id());
    resp_.set_bid_price(d.bid_price);
    resp_.set_win_probability(d.win_probability);
    resp_.set_expected_profit(d.expected_profit);
    resp_.set_used_fallback(used_fallback);
    resp_.set_inference_us(static_cast<int64_t>(secs * 1e6));
    responder_.Finish(resp_, Status::OK, this);
  }

  enum State { PROCESS, FINISH };
  AsyncBidServer* server_;
  ServerCompletionQueue* cq_;
  ServerContext ctx_;
  bidding::BidRequest req_;
  bidding::BidResponse resp_;
  ServerAsyncResponseWriter<bidding::BidResponse> responder_;
  State state_ = PROCESS;
  std::chrono::steady_clock::time_point t0_;
};

// ---- NotifyOutcome ----
class NotifyCall : public CallData {
 public:
  NotifyCall(AsyncBidServer* server, ServerCompletionQueue* cq)
      : server_(server), cq_(cq), responder_(&ctx_) {
    server_->service()->RequestNotifyOutcome(&ctx_, &req_, &responder_, cq_, cq_, this);
  }

  void Proceed(bool ok) override {
    if (state_ == PROCESS) {
      if (!ok) {
        delete this;
        return;
      }
      new NotifyCall(server_, cq_);
      server_->pipeline()->notify_outcome(req_.won(), req_.clearing_price(),
                                          req_.profit());
      resp_.set_ok(true);
      state_ = FINISH;
      responder_.Finish(resp_, Status::OK, this);
    } else {
      delete this;
    }
  }

 private:
  enum State { PROCESS, FINISH };
  AsyncBidServer* server_;
  ServerCompletionQueue* cq_;
  ServerContext ctx_;
  bidding::AuctionOutcome req_;
  bidding::Ack resp_;
  ServerAsyncResponseWriter<bidding::Ack> responder_;
  State state_ = PROCESS;
};

// ---- Check (health) ----
class CheckCall : public CallData {
 public:
  CheckCall(AsyncBidServer* server, ServerCompletionQueue* cq)
      : server_(server), cq_(cq), responder_(&ctx_) {
    server_->service()->RequestCheck(&ctx_, &req_, &responder_, cq_, cq_, this);
  }

  void Proceed(bool ok) override {
    if (state_ == PROCESS) {
      if (!ok) {
        delete this;
        return;
      }
      new CheckCall(server_, cq_);
      resp_.set_status(server_->pipeline()->ready()
                           ? bidding::HealthCheckResponse::SERVING
                           : bidding::HealthCheckResponse::NOT_SERVING);
      state_ = FINISH;
      responder_.Finish(resp_, Status::OK, this);
    } else {
      delete this;
    }
  }

 private:
  enum State { PROCESS, FINISH };
  AsyncBidServer* server_;
  ServerCompletionQueue* cq_;
  ServerContext ctx_;
  bidding::HealthCheckRequest req_;
  bidding::HealthCheckResponse resp_;
  ServerAsyncResponseWriter<bidding::HealthCheckResponse> responder_;
  State state_ = PROCESS;
};

AsyncBidServer::AsyncBidServer(const std::string& address, int cq_count,
                               BidPipeline* pipeline)
    : address_(address), cq_count_(cq_count < 1 ? 1 : cq_count),
      pipeline_(pipeline) {}

bool AsyncBidServer::start() {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);
  // HTTP/2 defaults to 100 concurrent streams per connection; the auction engine
  // and load tests keep many requests in flight on one connection, so raise it.
  builder.AddChannelArgument(GRPC_ARG_MAX_CONCURRENT_STREAMS, 4096);
  for (int i = 0; i < cq_count_; i++) {
    cqs_.push_back(builder.AddCompletionQueue());
  }
  server_ = builder.BuildAndStart();
  if (!server_) {
    return false;
  }
  // seed each completion queue with one outstanding acceptor per RPC type, then
  // start a polling thread to drive it
  for (auto& cq : cqs_) {
    new GetBidCall(this, cq.get());
    new NotifyCall(this, cq.get());
    new CheckCall(this, cq.get());
    grpc::ServerCompletionQueue* raw = cq.get();
    threads_.emplace_back([this, raw] { poll(raw); });
  }
  return true;
}

void AsyncBidServer::poll(ServerCompletionQueue* cq) {
  void* tag;
  bool ok;
  while (cq->Next(&tag, &ok)) {
    static_cast<CallData*>(tag)->Proceed(ok);
  }
}

void AsyncBidServer::stop(int deadline_ms) {
  auto deadline = std::chrono::system_clock::now() +
                  std::chrono::milliseconds(deadline_ms);
  if (server_) {
    server_->Shutdown(deadline);
  }
  // shutting down each CQ makes Next() drain then return false
  for (auto& cq : cqs_) {
    cq->Shutdown();
  }
  for (auto& t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

}  // namespace bidder
