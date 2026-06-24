#ifndef CPP_BIDDER_FALLBACK_H
#define CPP_BIDDER_FALLBACK_H

#include <string>
#include <unordered_map>

#include "types.h"

namespace bidder {

// A simple bid we use when the model isn't available (circuit breaker is open,
// or inference threw an error). The idea: a not-great bid that comes back on
// time still wins some auctions, which beats not bidding at all. We just take
// the advertiser's average past clearing price and shade it down a bit.
class Fallback {
 public:
  Fallback(float default_bid, float shading,
           std::unordered_map<std::string, float> by_advertiser)
      : default_bid_(default_bid),
        shading_(shading),
        by_advertiser_(std::move(by_advertiser)) {}

  // multiplier is the company's strategy multiplier (e.g. 1.2 for aggressive),
  // applied on top of the fallback bid.
  BidDecision bid_for(const AdRequest& req, float multiplier) const {
    float base = default_bid_;
    auto it = by_advertiser_.find(req.advertiser_id);
    if (it != by_advertiser_.end()) {
      base = it->second;
    }
    float bid = base * shading_ * multiplier;
    if (bid < 0.0f) {
      bid = 0.0f;
    }
    // never bid above the impression value
    if (req.impression_value > 0.0f && bid > req.impression_value) {
      bid = req.impression_value;
    }
    BidDecision d;
    d.bid_price = bid;
    d.win_probability = 0.0f;   // we don't know this without the model
    d.expected_profit = 0.0f;
    return d;
  }

 private:
  float default_bid_;
  float shading_;
  std::unordered_map<std::string, float> by_advertiser_;
};

}  // namespace bidder

#endif
