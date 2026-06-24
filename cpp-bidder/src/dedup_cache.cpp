#include "dedup_cache.h"

namespace bidder {

bool DedupCache::get(const std::string& id, BidDecision& out) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = map_.find(id);
  if (it == map_.end()) {
    return false;
  }
  auto now = std::chrono::steady_clock::now();
  if (now >= it->second.expiry) {
    // stale: drop it
    lru_.erase(it->second.lru_it);
    map_.erase(it);
    return false;
  }
  // move to front (most recently used)
  lru_.erase(it->second.lru_it);
  lru_.push_front(id);
  it->second.lru_it = lru_.begin();
  out = it->second.decision;
  return true;
}

void DedupCache::put(const std::string& id, const BidDecision& decision) {
  std::lock_guard<std::mutex> lk(mu_);
  auto now = std::chrono::steady_clock::now();
  auto expiry = now + std::chrono::milliseconds(ttl_ms_);

  auto it = map_.find(id);
  if (it != map_.end()) {
    it->second.decision = decision;
    it->second.expiry = expiry;
    lru_.erase(it->second.lru_it);
    lru_.push_front(id);
    it->second.lru_it = lru_.begin();
    return;
  }

  // evict the least recently used entries while at capacity
  while (static_cast<int>(map_.size()) >= max_size_ && !lru_.empty()) {
    const std::string& victim = lru_.back();
    map_.erase(victim);
    lru_.pop_back();
  }

  lru_.push_front(id);
  Entry e;
  e.decision = decision;
  e.expiry = expiry;
  e.lru_it = lru_.begin();
  map_[id] = e;
}

}  // namespace bidder
