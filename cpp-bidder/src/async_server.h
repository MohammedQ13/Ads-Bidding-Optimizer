#ifndef CPP_BIDDER_ASYNC_SERVER_H
#define CPP_BIDDER_ASYNC_SERVER_H

#include <memory>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "bid_service.grpc.pb.h"

#include "bid_pipeline.h"

namespace bidder {

// Asynchronous gRPC server built on completion queues. A small fixed set of CQ
// polling threads accept RPCs and never block: each GetBid hands off to the
// (non-blocking) BidPipeline and the response is sent from the batcher's
// callback when inference finishes. This decouples in-flight request count from
// thread count, so the server can hold thousands of requests in flight on a
// handful of threads - which is what exposes the engine's real throughput end
// to end (the sync server caps out because it holds a thread per in-flight RPC).
class AsyncBidServer {
 public:
  AsyncBidServer(const std::string& address, int cq_count, BidPipeline* pipeline);

  // Build and start the server and the CQ polling threads. Returns false if the
  // server failed to bind.
  bool start();

  // Drain and stop (graceful). deadline_ms bounds the drain.
  void stop(int deadline_ms);

  bidding::BidService::AsyncService* service() { return &service_; }
  BidPipeline* pipeline() { return pipeline_; }

 private:
  void poll(grpc::ServerCompletionQueue* cq);

  std::string address_;
  int cq_count_;
  BidPipeline* pipeline_;

  bidding::BidService::AsyncService service_;
  std::unique_ptr<grpc::Server> server_;
  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> cqs_;
  std::vector<std::thread> threads_;
};

}  // namespace bidder

#endif
