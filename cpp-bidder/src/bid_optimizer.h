#ifndef CPP_BIDDER_BID_OPTIMIZER_H
#define CPP_BIDDER_BID_OPTIMIZER_H

#include "types.h"

namespace bidder {

// Turns the model's predicted price distribution into a bid. The model gives a
// probability for each price bin, and bin k just means a price of k fen. So the
// CDF (the chance the price is <= b, which is also our chance of winning if we
// bid b) is the running sum of the probabilities, and the expected profit if we
// bid b is (V - b) * CDF(b). We try every bin and pick the most profitable bid.
// This is the same thing the Python grid_optimize_bins does, which is where the
// regret numbers came from.
//
// probs points to num_bins probabilities (the ONNX graph already did the
// softmax). V is what the impression is worth. multiplier scales the bid for a
// company's strategy (1.0 = profit maximizer, >1 aggressive, <1 conservative).
BidDecision optimize_bid(const float* probs, int num_bins, float V,
                         float multiplier = 1.0f);

}  // namespace bidder

#endif
