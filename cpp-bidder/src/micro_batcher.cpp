#include "micro_batcher.h"

#include <chrono>

namespace bidder {

MicroBatcher::MicroBatcher(ModelSession& model, int num_workers, int batch_max,
                           int queue_max, StatsHook hook)
    : model_(model),
      batch_max_(batch_max),
      queue_max_(queue_max),
      hook_(std::move(hook)) {
  if (num_workers < 1) {
    num_workers = 1;
  }
  for (int i = 0; i < num_workers; i++) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

MicroBatcher::~MicroBatcher() { stop(); }

void MicroBatcher::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

int MicroBatcher::queue_depth() {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<int>(queue_.size());
}

bool MicroBatcher::submit(const EncodedFeatures& feat, ResultCb cb) {
  std::unique_lock<std::mutex> lk(mu_);
  if (stopping_ || static_cast<int>(queue_.size()) >= queue_max_) {
    return false;  // shed load: queue full
  }
  queue_.push_back(Job{feat, std::move(cb)});
  lk.unlock();
  cv_.notify_one();
  return true;
}

void MicroBatcher::worker_loop() {
  while (true) {
    std::vector<Job> batch;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_ && queue_.empty()) {
        return;
      }
      // grab whatever is waiting, up to the batch cap
      int take = static_cast<int>(queue_.size());
      if (take > batch_max_) {
        take = batch_max_;
      }
      batch.reserve(take);
      for (int i = 0; i < take; i++) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }
    if (!batch.empty()) {
      run_batch(batch);
    }
  }
}

void MicroBatcher::run_batch(std::vector<Job>& batch) {
  int B = static_cast<int>(batch.size());
  std::vector<int64_t> cat(static_cast<size_t>(B) * 9);
  std::vector<float> cont(static_cast<size_t>(B) * 9);
  std::vector<int64_t> tags(static_cast<size_t>(B) * 10);
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < 9; j++) {
      cat[i * 9 + j] = batch[i].feat.cat[j];
      cont[i * 9 + j] = batch[i].feat.cont[j];
    }
    for (int j = 0; j < 10; j++) {
      tags[i * 10 + j] = batch[i].feat.tags[j];
    }
  }

  std::vector<float> probs;
  auto t0 = std::chrono::steady_clock::now();
  try {
    model_.run(cat.data(), cont.data(), tags.data(), B, probs);
  } catch (...) {
    // inference threw: tell every request in the batch so each can fall back
    std::vector<float> empty;
    for (auto& job : batch) {
      job.cb(false, empty);
    }
    return;
  }
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  if (hook_) {
    hook_(B, secs);
  }

  int num_bins = model_.num_bins();
  for (int i = 0; i < B; i++) {
    std::vector<float> one(num_bins);
    for (int k = 0; k < num_bins; k++) {
      one[k] = probs[static_cast<size_t>(i) * num_bins + k];
    }
    batch[i].cb(true, one);
  }
}

}  // namespace bidder
