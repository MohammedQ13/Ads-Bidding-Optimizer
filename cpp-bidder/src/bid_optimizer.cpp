#include "bid_optimizer.h"

namespace bidder {

// CDF (cumulative probability) up to a bid that might land between two integer
// bins, using a straight-line guess between the two. We need this to report the
// win probability at the final bid after the strategy multiplier is applied.
static double cdf_at(const float* probs, int num_bins, double bid) {
  if (bid <= 0.0) {
    return 0.0;
  }
  int lo = static_cast<int>(bid);
  if (lo >= num_bins - 1) {
    double total = 0.0;
    for (int k = 0; k < num_bins; k++) {
      total += probs[k];
    }
    return total;
  }
  double cum = 0.0;
  for (int k = 0; k <= lo; k++) {
    cum += probs[k];
  }
  double frac = bid - lo;
  return cum + frac * probs[lo + 1];
}

BidDecision optimize_bid(const float* probs, int num_bins, float V,
                         float multiplier) {
  // Try every bin. The bin index is the price in fen, profit = (V - b) * CDF(b).
  // Bidding more than V is never worth it, so we force that profit to zero. If
  // two bins tie, we keep the first one (same as numpy argmax).
  double running = 0.0;
  double best_profit = -1e18;
  int best_k = 0;
  double best_cdf = 0.0;
  for (int k = 0; k < num_bins; k++) {
    running += probs[k];
    double profit;
    if (static_cast<double>(k) > V) {
      profit = 0.0;
    } else {
      profit = (V - static_cast<double>(k)) * running;
    }
    if (profit > best_profit) {
      best_profit = profit;
      best_k = k;
      best_cdf = running;
    }
  }

  BidDecision out;
  if (best_profit < 0.0) {
    // no profitable bid, sit this auction out
    out.bid_price = 0.0f;
    out.win_probability = 0.0f;
    out.expected_profit = 0.0f;
    return out;
  }

  // apply the company's strategy multiplier and clamp to the valid price range
  double bid = static_cast<double>(best_k) * static_cast<double>(multiplier);
  if (bid < 0.0) {
    bid = 0.0;
  }
  double max_bid = num_bins - 1;
  if (bid > max_bid) {
    bid = max_bid;
  }

  double win_p;
  if (multiplier == 1.0f) {
    win_p = best_cdf;  // exact, avoids a second pass on the parity path
  } else {
    win_p = cdf_at(probs, num_bins, bid);
  }

  out.bid_price = static_cast<float>(bid);
  out.win_probability = static_cast<float>(win_p);
  out.expected_profit = static_cast<float>((static_cast<double>(V) - bid) * win_p);
  return out;
}

}  // namespace bidder
