#ifndef CPP_BIDDER_CIRCUIT_BREAKER_H
#define CPP_BIDDER_CIRCUIT_BREAKER_H

#include <atomic>
#include <chrono>
#include <mutex>

namespace bidder {

// Stops calling the model if it keeps failing, so one broken model doesn't turn
// into a flood of timeouts. While it's "open" we just use the fallback bid.
// After a cooldown we let one request through ("half-open") to check if the
// model works again. This is a common pattern from microservices.
class CircuitBreaker {
 public:
  enum class State { CLOSED = 0, OPEN = 1, HALF_OPEN = 2 };

  CircuitBreaker(int failure_threshold, int cooldown_ms)
      : failure_threshold_(failure_threshold), cooldown_ms_(cooldown_ms) {}

  // Returns true if a request is allowed to use the model right now. If we're
  // open but the cooldown is over, we flip to half-open and let this one through
  // to test the model.
  bool allow_request();

  void record_success();
  void record_failure();

  // 0 = closed, 1 = open, 2 = half-open. Used for the metrics gauge.
  int state_code() const { return static_cast<int>(state_.load()); }

 private:
  std::mutex mu_;
  std::atomic<State> state_{State::CLOSED};
  int failure_threshold_;
  int cooldown_ms_;
  int consecutive_failures_ = 0;
  std::chrono::steady_clock::time_point opened_at_;
};

}  // namespace bidder

#endif
