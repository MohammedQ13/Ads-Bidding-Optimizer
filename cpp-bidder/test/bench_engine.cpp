// In-process throughput of the core bid engine: feature encoding + ONNX
// inference + bid optimization, with no gRPC. This isolates the compute capacity
// from the network/serving layer, so it answers "how many bids can the engine
// evaluate per second on N cores?" - the honest compute ceiling. Each worker
// runs batched inference the same way the server's micro-batcher does.
//
// usage: bench_engine <threads> <batch> <seconds> [config] [model]

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "../src/feature_store.h"
#include "../src/model_session.h"
#include "../src/bid_optimizer.h"

using namespace bidder;

static AdRequest sample_request() {
  AdRequest r;
  r.request_id = "x";
  r.region = 216;
  r.city = 1;
  r.domain = "trqRTuiEMaYZ";
  r.ad_exchange = 1;
  r.slot_width = 300;
  r.slot_height = 250;
  r.slot_visibility = 1;
  r.slot_format = 0;
  r.advertiser_id = "1458";
  r.user_tags = {"10063", "10024"};
  r.slot_floor_price = 0.0f;
  r.impression_value = 150.0f;
  r.timestamp = 20130606120000LL;
  return r;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? std::stoi(argv[1]) : 6;
  int batch = argc > 2 ? std::stoi(argv[2]) : 32;
  int seconds = argc > 3 ? std::stoi(argv[3]) : 10;
  std::string config = argc > 4 ? argv[4] : "models/feature_config.json";
  std::string model = argc > 5 ? argv[5] : "models/bid_model.onnx";

  FeatureStore store(config);
  ModelSession session(model);
  int num_bins = session.num_bins();

  std::atomic<bool> stop{false};
  std::atomic<long> total{0};

  auto worker = [&]() {
    AdRequest req = sample_request();
    // pre-encode a batch of features (encoding is per-request but identical here)
    std::vector<int64_t> cat(batch * 9);
    std::vector<float> cont(batch * 9);
    std::vector<int64_t> tags(batch * 10);
    std::vector<float> probs;
    long local = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      for (int i = 0; i < batch; i++) {
        EncodedFeatures e = store.encode(req);
        for (int j = 0; j < 9; j++) {
          cat[i * 9 + j] = e.cat[j];
          cont[i * 9 + j] = e.cont[j];
        }
        for (int j = 0; j < 10; j++) tags[i * 10 + j] = e.tags[j];
      }
      session.run(cat.data(), cont.data(), tags.data(), batch, probs);
      for (int i = 0; i < batch; i++) {
        optimize_bid(probs.data() + (size_t)i * num_bins, num_bins, req.impression_value, 1.0f);
      }
      local += batch;
    }
    total.fetch_add(local, std::memory_order_relaxed);
  };

  // warm up
  {
    AdRequest req = sample_request();
    std::vector<int64_t> cat(9), tags(10);
    std::vector<float> cont(9), probs;
    EncodedFeatures e = store.encode(req);
    for (int j = 0; j < 9; j++) { cat[j] = e.cat[j]; cont[j] = e.cont[j]; }
    for (int j = 0; j < 10; j++) tags[j] = e.tags[j];
    for (int i = 0; i < 2000; i++) session.run(cat.data(), cont.data(), tags.data(), 1, probs);
  }

  std::printf("engine throughput: %d threads, batch %d, %ds\n", threads, batch, seconds);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) pool.emplace_back(worker);
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  stop.store(true);
  for (auto& th : pool) th.join();
  auto t1 = std::chrono::steady_clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  long n = total.load();
  std::printf("evaluated %ld bids in %.2fs\n", n, secs);
  std::printf("throughput: %.0f bids/sec\n", n / secs);
  std::printf("per-bid: %.2f us\n", secs * 1e6 / n);
  return 0;
}
