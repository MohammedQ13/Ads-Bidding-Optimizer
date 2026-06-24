#include "circuit_breaker.h"

namespace bidder {

bool CircuitBreaker::allow_request() {
  std::lock_guard<std::mutex> lk(mu_);
  if (state_.load() == State::OPEN) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - opened_at_)
                       .count();
    if (elapsed >= cooldown_ms_) {
      // cooldown is over, let one request through to see if the model recovered
      state_.store(State::HALF_OPEN);
      return true;
    }
    return false;
  }
  // closed or half-open: allow
  return true;
}

void CircuitBreaker::record_success() {
  std::lock_guard<std::mutex> lk(mu_);
  consecutive_failures_ = 0;
  state_.store(State::CLOSED);
}

void CircuitBreaker::record_failure() {
  std::lock_guard<std::mutex> lk(mu_);
  if (state_.load() == State::HALF_OPEN) {
    // probe failed, open again and restart the cooldown
    state_.store(State::OPEN);
    opened_at_ = std::chrono::steady_clock::now();
    return;
  }
  consecutive_failures_++;
  if (consecutive_failures_ >= failure_threshold_) {
    state_.store(State::OPEN);
    opened_at_ = std::chrono::steady_clock::now();
  }
}

}  // namespace bidder
