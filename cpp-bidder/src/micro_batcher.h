#ifndef CPP_BIDDER_MICRO_BATCHER_H
#define CPP_BIDDER_MICRO_BATCHER_H

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "types.h"
#include "model_session.h"

namespace bidder {

// Groups individual requests into batches before running the model. Each ONNX
// Run() call costs about the same no matter how many requests are in it, so
// putting more requests in one call gives a lot more throughput for almost the
// same cost. Worker threads grab whatever requests are waiting and run them
// together.
//
// This is callback-based, not future-based: when a request's batch finishes,
// the worker calls the request's callback directly. That is what lets the async
// gRPC server avoid blocking a thread per in-flight request - the callback just
// fills the response and sends it. The queue is bounded; submit() returns false
// when it is full so the server can shed load.
class MicroBatcher {
 public:
  // Called when a request's result is ready. ok=false means inference threw, so
  // the caller should fall back. probs is only valid when ok=true.
  using ResultCb = std::function<void(bool ok, std::vector<float>& probs)>;

  // Called after each batch runs, with the batch size and inference seconds.
  using StatsHook = std::function<void(int batch_size, double infer_seconds)>;

  MicroBatcher(ModelSession& model, int num_workers, int batch_max,
               int queue_max, StatsHook hook);
  ~MicroBatcher();

  // Queue a request. Returns false if the queue is full (shed load). On success
  // the callback fires later on a worker thread when the batch completes.
  bool submit(const EncodedFeatures& feat, ResultCb cb);

  int queue_depth();
  void stop();

 private:
  struct Job {
    EncodedFeatures feat;
    ResultCb cb;
  };

  void worker_loop();
  void run_batch(std::vector<Job>& batch);

  ModelSession& model_;
  int batch_max_;
  int queue_max_;
  StatsHook hook_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  bool stopping_ = false;

  std::vector<std::thread> workers_;
};

}  // namespace bidder

#endif
