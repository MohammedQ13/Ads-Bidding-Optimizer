#ifndef CPP_BIDDER_TYPES_H
#define CPP_BIDDER_TYPES_H

#include <string>
#include <vector>
#include <cstdint>

namespace bidder {

// One ad auction we might bid on. We keep this separate from the protobuf
// BidRequest so we can test the bidding logic without needing gRPC. The gRPC
// code just copies a BidRequest into one of these.
struct AdRequest {
  std::string request_id;
  int region = 0;
  int city = 0;
  std::string domain;
  int ad_exchange = 0;
  int slot_width = 0;
  int slot_height = 0;
  int slot_visibility = 0;
  int slot_format = 0;
  std::string advertiser_id;
  std::vector<std::string> user_tags;  // already split into individual tokens
  float slot_floor_price = 0.0f;
  float impression_value = 0.0f;       // V, set by the auction engine
  int64_t timestamp = 0;               // YYYYMMDDHHmmss
};

// The model inputs for one request, already turned into numbers. The sizes are
// fixed to match the ONNX model: 9 category ids, 9 continuous values, 10 tag ids.
struct EncodedFeatures {
  int64_t cat[9];
  float cont[9];
  int64_t tags[10];
};

// The bid we decided on for one request.
struct BidDecision {
  float bid_price = 0.0f;
  float win_probability = 0.0f;
  float expected_profit = 0.0f;
};

}  // namespace bidder

#endif
