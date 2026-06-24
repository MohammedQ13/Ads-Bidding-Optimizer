// Entry point for the C++ bid server. Loads config, builds the bid pipeline,
// starts the async (completion-queue) gRPC server and the Prometheus endpoint,
// and handles graceful shutdown.

#include <atomic>
#include <csignal>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>

#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "config.h"
#include "metrics.h"
#include "bid_pipeline.h"
#include "async_server.h"

using namespace bidder;

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }
}  // namespace

// Watches the model file and reloads it whenever its modified-time changes.
// This is how hot-reload works in practice: the retrainer writes a new ONNX
// file and the server notices it and swaps it in without going down.
static void watch_model_loop(BidPipeline* pipeline, const std::string& path,
                             int interval_ms) {
  std::error_code ec;
  auto last = std::filesystem::last_write_time(path, ec);
  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    std::error_code e;
    auto cur = std::filesystem::last_write_time(path, e);
    if (!e && cur != last) {
      last = cur;
      std::cout << "model file changed, reloading: " << path << std::endl;
      if (pipeline->reload_model(path)) {
        std::cout << "model reloaded ok" << std::endl;
      } else {
        std::cerr << "model reload rejected (validation failed)" << std::endl;
      }
    }
  }
}

int main(int argc, char** argv) {
  std::string config_path = "config/config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }
  ServerConfig cfg = ServerConfig::load(config_path);

  std::cout << "starting DSP '" << cfg.dsp_id << "' strategy '"
            << cfg.strategy_type << "' multiplier " << cfg.bid_multiplier
            << std::endl;

  std::unique_ptr<Metrics> metrics;
  try {
    std::string bind = "0.0.0.0:" + std::to_string(cfg.metrics_port);
    metrics = std::make_unique<Metrics>(bind, cfg.dsp_id);
    std::cout << "metrics on :" << cfg.metrics_port << "/metrics" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "metrics disabled: " << e.what() << std::endl;
  }

  std::unique_ptr<BidPipeline> pipeline;
  try {
    pipeline = std::make_unique<BidPipeline>(cfg, metrics.get());
  } catch (const std::exception& e) {
    std::cerr << "failed to build pipeline: " << e.what() << std::endl;
    return 1;
  }

  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  AsyncBidServer server(cfg.grpc_address, cfg.cq_threads, pipeline.get());
  if (!server.start()) {
    std::cerr << "failed to start gRPC server on " << cfg.grpc_address
              << std::endl;
    return 1;
  }
  std::cout << "async bid server listening on " << cfg.grpc_address << " ("
            << cfg.cq_threads << " CQ threads, " << cfg.batch_workers
            << " batch workers)" << std::endl;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::thread watcher;
  if (cfg.watch_model) {
    watcher = std::thread(watch_model_loop, pipeline.get(), cfg.model_path,
                          cfg.watch_interval_ms);
  }

  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::cout << "shutting down: draining in-flight requests" << std::endl;
  pipeline->set_ready(false);  // health check now reports NOT_SERVING
  server.stop(3000);
  if (watcher.joinable()) {
    watcher.join();
  }
  std::cout << "stopped" << std::endl;
  return 0;
}
