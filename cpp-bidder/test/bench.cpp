// Simple closed-loop load generator for the bid server. Fires a fixed number of
// GetBid requests across N concurrent client threads and reports throughput and
// the latency distribution (p50/p95/p99/max). Warms up first so ONNX JIT and
// the allocator arenas are hot, per the benchmarking notes in cpp_server.md.
//
// Closed-loop (each thread waits for a reply before sending the next) so the
// p99 here is optimistic under saturation - good enough to find the knee.
//
// usage: bench <target> <threads> <requests_per_thread>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "bid_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using namespace bidding;

static void make_request(BidRequest& req, int i) {
  req.set_request_id("bench-" + std::to_string(i));
  req.set_region(216);
  req.set_city(1);
  req.set_domain("trqRTuiEMaYZ");
  req.set_ad_exchange(1);
  req.set_slot_width(300);
  req.set_slot_height(250);
  req.set_slot_visibility(1);
  req.set_slot_format(0);
  req.set_advertiser_id("1458");
  req.add_user_tags("10063,10024,13403");
  req.set_slot_floor_price(0.0f);
  req.set_impression_value(150.0f);
  req.set_timestamp(20130606120000LL);
}

int main(int argc, char** argv) {
  std::string target = argc > 1 ? argv[1] : "localhost:50051";
  int threads = argc > 2 ? std::stoi(argv[2]) : 50;
  int per_thread = argc > 3 ? std::stoi(argv[3]) : 2000;

  auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  auto stub = BidService::NewStub(channel);
  if (!channel->WaitForConnected(std::chrono::system_clock::now() +
                                 std::chrono::seconds(5))) {
    std::printf("could not connect to %s\n", target.c_str());
    return 2;
  }

  // warm up
  for (int i = 0; i < 10000; i++) {
    BidRequest req;
    make_request(req, i);
    req.set_request_id("warm-" + std::to_string(i));  // avoid dedup cache
    ClientContext ctx;
    BidResponse resp;
    stub->GetBid(&ctx, req, &resp);
  }

  std::vector<std::vector<double>> lat(threads);
  std::atomic<long> errors{0};

  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) {
    pool.emplace_back([&, t] {
      lat[t].reserve(per_thread);
      for (int i = 0; i < per_thread; i++) {
        BidRequest req;
        make_request(req, t * per_thread + i);
        ClientContext ctx;
        BidResponse resp;
        auto s0 = std::chrono::steady_clock::now();
        auto st = stub->GetBid(&ctx, req, &resp);
        auto s1 = std::chrono::steady_clock::now();
        if (!st.ok()) {
          errors.fetch_add(1);
        }
        lat[t].push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }
  auto t1 = std::chrono::steady_clock::now();

  std::vector<double> all;
  all.reserve(static_cast<size_t>(threads) * per_thread);
  for (auto& v : lat) {
    for (double x : v) {
      all.push_back(x);
    }
  }
  std::sort(all.begin(), all.end());
  auto pct = [&](double p) {
    if (all.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * (all.size() - 1));
    return all[idx];
  };

  double secs = std::chrono::duration<double>(t1 - t0).count();
  long total = static_cast<long>(all.size());
  std::printf("threads=%d total=%ld errors=%ld\n", threads, total, errors.load());
  std::printf("throughput: %.0f req/s\n", total / secs);
  std::printf("latency ms: p50=%.3f p95=%.3f p99=%.3f p999=%.3f max=%.3f\n",
              pct(0.50), pct(0.95), pct(0.99), pct(0.999), pct(1.0));
  return 0;
}
