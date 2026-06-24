#ifndef CPP_BIDDER_DEDUP_CACHE_H
#define CPP_BIDDER_DEDUP_CACHE_H

#include <chrono>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "types.h"

namespace bidder {

// A small short-lived cache keyed on request_id. If the auction engine sends the
// same request again within the TTL (networks retry), we hand back the bid we
// already computed instead of doing the work twice. It's an LRU cache with a
// fixed size so it can't grow forever, and entries also time out.
class DedupCache {
 public:
  DedupCache(int max_size, int ttl_ms)
      : max_size_(max_size), ttl_ms_(ttl_ms) {}

  // If we have a fresh (not expired) entry for this id, fill out and return true.
  bool get(const std::string& id, BidDecision& out);

  void put(const std::string& id, const BidDecision& decision);

 private:
  struct Entry {
    BidDecision decision;
    std::chrono::steady_clock::time_point expiry;
    std::list<std::string>::iterator lru_it;
  };

  std::mutex mu_;
  int max_size_;
  int ttl_ms_;
  std::unordered_map<std::string, Entry> map_;
  std::list<std::string> lru_;  // front = most recently used
};

}  // namespace bidder

#endif
