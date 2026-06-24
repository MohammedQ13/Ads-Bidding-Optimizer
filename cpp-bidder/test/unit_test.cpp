// Unit tests for the standalone C++ components: the circuit breaker state
// machine, the dedup cache (TTL + LRU eviction), and the bid optimizer edge
// cases. No gRPC or ONNX needed - these run fast and deterministically.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "../src/circuit_breaker.h"
#include "../src/dedup_cache.h"
#include "../src/bid_optimizer.h"

using namespace bidder;

static int failures = 0;
static void check(bool ok, const std::string& msg) {
  if (!ok) {
    failures++;
    std::printf("  FAIL: %s\n", msg.c_str());
  }
}

static void test_circuit_breaker() {
  // threshold 3, cooldown 50ms
  CircuitBreaker cb(3, 50);
  check(cb.allow_request(), "breaker starts closed and allows");
  check(cb.state_code() == 0, "starts in CLOSED");

  cb.record_failure();
  cb.record_failure();
  check(cb.allow_request(), "still closed after 2 of 3 failures");
  cb.record_failure();  // 3rd failure -> open
  check(cb.state_code() == 1, "OPEN after threshold failures");
  check(!cb.allow_request(), "open breaker blocks requests");

  // a success in CLOSED resets the failure count
  CircuitBreaker cb2(3, 50);
  cb2.record_failure();
  cb2.record_failure();
  cb2.record_success();
  cb2.record_failure();
  cb2.record_failure();
  check(cb2.allow_request(), "success reset the counter, still closed");

  // after cooldown, an open breaker goes half-open and lets one through
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  check(cb.allow_request(), "after cooldown the open breaker allows a probe");
  check(cb.state_code() == 2, "HALF_OPEN after cooldown probe");
  cb.record_success();
  check(cb.state_code() == 0, "success in half-open closes the breaker");
}

static void test_dedup_cache() {
  DedupCache cache(2, 50);  // size 2, TTL 50ms
  BidDecision d;
  d.bid_price = 42.0f;
  cache.put("r1", d);

  BidDecision got;
  check(cache.get("r1", got), "fresh entry is found");
  check(got.bid_price == 42.0f, "cached value is correct");
  check(!cache.get("missing", got), "unknown id is a miss");

  // TTL expiry
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  check(!cache.get("r1", got), "entry expires after the TTL");

  // LRU eviction at capacity 2
  DedupCache lru(2, 10000);
  BidDecision a, b, c;
  a.bid_price = 1;
  b.bid_price = 2;
  c.bid_price = 3;
  lru.put("a", a);
  lru.put("b", b);
  lru.get("a", got);   // touch a so b becomes least recently used
  lru.put("c", c);     // should evict b
  check(lru.get("a", got), "a survives (recently used)");
  check(lru.get("c", got), "c was just inserted");
  check(!lru.get("b", got), "b was evicted as least recently used");
}

static void test_bid_optimizer() {
  int n = 200;
  std::vector<float> probs(n, 0.0f);
  probs[50] = 1.0f;  // all mass at price 50

  // V well above the peak: best bid is exactly 50, profit (150-50)*1 = 100
  BidDecision d = optimize_bid(probs.data(), n, 150.0f, 1.0f);
  check(d.bid_price == 50.0f, "bids the peak when V is high");
  check(d.win_probability > 0.99f, "win prob ~1 at the peak");
  check(d.expected_profit > 99.0f && d.expected_profit < 101.0f, "profit ~100");

  // V below the peak: cannot win cheaply, should bid 0
  BidDecision d2 = optimize_bid(probs.data(), n, 30.0f, 1.0f);
  check(d2.bid_price == 0.0f, "bids 0 when the price is above V");

  // strategy multiplier scales the bid (1.2x of 50 = 60). use a tolerance since
  // 50 * 1.2f is not exactly 60 in float.
  BidDecision d3 = optimize_bid(probs.data(), n, 150.0f, 1.2f);
  check(std::fabs(d3.bid_price - 60.0f) < 1e-3f, "multiplier scales the bid (50 * 1.2 = 60)");

  // spread distribution: mass split between 40 and 80
  std::vector<float> probs2(n, 0.0f);
  probs2[40] = 0.5f;
  probs2[80] = 0.5f;
  BidDecision d4 = optimize_bid(probs2.data(), n, 150.0f, 1.0f);
  // bidding 40 wins w.p. 0.5 -> (150-40)*0.5 = 55; bidding 80 wins w.p. 1 ->
  // (150-80)*1 = 70. So 80 is better.
  check(d4.bid_price == 80.0f, "picks the higher-profit bin on a spread dist");
}

int main() {
  test_circuit_breaker();
  test_dedup_cache();
  test_bid_optimizer();
  if (failures == 0) {
    std::printf("UNIT TESTS OK\n");
    return 0;
  }
  std::printf("UNIT TESTS FAILED: %d checks\n", failures);
  return 1;
}
