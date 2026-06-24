// Async (open-loop) gRPC load generator. Unlike bench.cpp (closed-loop, one
// outstanding request per thread), this keeps a large, fixed number of requests
// in flight at all times using gRPC completion queues. That actually saturates
// the server - so the throughput number reflects the server's capacity (and
// fills the micro-batcher), not the client's thread count. It also exercises the
// async gRPC client pattern.
//
// usage: bench_async <target> <threads> <inflight_per_thread> <seconds>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "bid_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using namespace bidding;

static void fill(BidRequest& r, long i) {
  r.set_request_id("b" + std::to_string(i));
  r.set_region(216);
  r.set_city(1);
  r.set_domain("trqRTuiEMaYZ");
  r.set_ad_exchange(1);
  r.set_slot_width(300);
  r.set_slot_height(250);
  r.set_slot_visibility(1);
  r.set_slot_format(0);
  r.set_advertiser_id("1458");
  r.add_user_tags("10063,10024");
  r.set_slot_floor_price(0.0f);
  r.set_impression_value(150.0f);
  r.set_timestamp(20130606120000LL);
}

// one outstanding async call
struct Call {
  ClientContext ctx;
  BidResponse resp;
  Status status;
  std::unique_ptr<ClientAsyncResponseReader<BidResponse>> reader;
  std::chrono::steady_clock::time_point sent;
};

int main(int argc, char** argv) {
  std::string target = argc > 1 ? argv[1] : "localhost:50051";
  int threads = argc > 2 ? std::stoi(argv[2]) : 4;
  int inflight = argc > 3 ? std::stoi(argv[3]) : 512;
  int seconds = argc > 4 ? std::stoi(argv[4]) : 10;

  // one channel PER thread: a single HTTP/2 connection caps concurrent streams
  // (and its framing is single-threaded), so sharing one channel would bottleneck
  // the client, not measure the server. Each thread gets its own connection.
  std::vector<std::shared_ptr<grpc::Channel>> channels(threads);
  for (int t = 0; t < threads; t++) {
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 4096);
    // a unique arg per channel prevents gRPC from sharing one subchannel
    args.SetInt("bench.channel_id", t);
    channels[t] = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
  }
  auto stub0 = BidService::NewStub(channels[0]);
  if (!channels[0]->WaitForConnected(std::chrono::system_clock::now() +
                                     std::chrono::seconds(5))) {
    std::printf("could not connect to %s\n", target.c_str());
    return 2;
  }

  std::atomic<bool> stop{false};
  std::atomic<long> total{0};
  std::atomic<long> errors{0};
  std::vector<std::vector<double>> lat(threads);

  auto worker = [&](int t) {
    auto stub = BidService::NewStub(channels[t]);
    CompletionQueue cq;
    long seq = static_cast<long>(t) << 40;
    int outstanding = 0;
    lat[t].reserve(2000000);

    auto fire = [&]() {
      BidRequest req;
      fill(req, seq++);
      Call* c = new Call();
      c->sent = std::chrono::steady_clock::now();
      c->reader = stub->AsyncGetBid(&c->ctx, req, &cq);
      c->reader->Finish(&c->resp, &c->status, (void*)c);
      outstanding++;
    };

    for (int i = 0; i < inflight; i++) fire();

    void* tag;
    bool ok;
    while (outstanding > 0 && cq.Next(&tag, &ok)) {
      Call* c = static_cast<Call*>(tag);
      auto now = std::chrono::steady_clock::now();
      if (!ok || !c->status.ok()) {
        errors.fetch_add(1, std::memory_order_relaxed);
      } else {
        lat[t].push_back(std::chrono::duration<double, std::milli>(now - c->sent).count());
        total.fetch_add(1, std::memory_order_relaxed);
      }
      delete c;
      outstanding--;
      if (!stop.load(std::memory_order_relaxed)) {
        fire();
      }
    }
    cq.Shutdown();
  };

  // warm up briefly
  std::printf("warming up...\n");
  {
    std::atomic<bool> wstop{false};
    std::thread w([&] {
      CompletionQueue cq;
      int out = 0;
      long wseq = 1;
      auto fire = [&]() {
        BidRequest req; fill(req, wseq++);  // unique ids so warmup doesn't all cache-hit
        Call* c = new Call(); c->sent = std::chrono::steady_clock::now();
        c->reader = stub0->AsyncGetBid(&c->ctx, req, &cq);
        c->reader->Finish(&c->resp, &c->status, (void*)c); out++;
      };
      for (int i = 0; i < 256; i++) fire();
      void* tag; bool ok;
      auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (out > 0 && cq.Next(&tag, &ok)) {
        delete static_cast<Call*>(tag); out--;
        if (std::chrono::steady_clock::now() < until) fire();
      }
      cq.Shutdown();
    });
    w.join();
  }

  std::printf("load: %d threads x %d in-flight for %ds\n", threads, inflight, seconds);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  stop.store(true);
  for (auto& th : pool) th.join();
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();

  std::vector<double> all;
  for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  auto pct = [&](double p) {
    if (all.empty()) return 0.0;
    return all[std::min(all.size() - 1, (size_t)(p * all.size()))];
  };

  long n = total.load();
  std::printf("\n=== async load test ===\n");
  std::printf("completed: %ld   errors: %ld   wall: %.2fs\n", n, errors.load(), secs);
  std::printf("throughput: %.0f req/s\n", n / secs);
  std::printf("latency ms: p50=%.2f p95=%.2f p99=%.2f p999=%.2f max=%.2f\n",
              pct(0.50), pct(0.95), pct(0.99), pct(0.999), pct(0.9999));
  return 0;
}
