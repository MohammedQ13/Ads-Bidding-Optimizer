#ifndef CPP_BIDDER_MODEL_SESSION_H
#define CPP_BIDDER_MODEL_SESSION_H

#include <string>
#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>

#include <onnxruntime_cxx_api.h>

namespace bidder {

// Wraps an ONNX Runtime session and runs the bid model. The session sits behind
// a shared_ptr we can swap atomically. That swap is the hook Phase 3's
// hot-reload uses: a background thread builds a new session, checks it, then
// swaps it in while requests that are already running keep using the old one
// until they finish.
//
// The model is tiny, so we tell ONNX to run it single-threaded
// (intra_op = inter_op = 1) and get our speed from lots of worker threads
// calling Run() at the same time instead. ONNX sessions are safe to call Run()
// on from many threads at once.
class ModelSession {
 public:
  explicit ModelSession(const std::string& model_path);

  // Load a new model file, check it works, and swap it in. If loading or the
  // check fails it throws, and the model currently serving is left alone.
  void reload(const std::string& model_path);

  // Run a batch through the model. cat is batch*9 int64, cont is batch*9 float,
  // tags is batch*10 int64. out_probs is resized to batch*num_bins.
  void run(const int64_t* cat, const float* cont, const int64_t* tags,
           int batch, std::vector<float>& out_probs);

  int num_bins() const { return num_bins_; }

 private:
  std::shared_ptr<Ort::Session> build(const std::string& model_path);
  std::shared_ptr<Ort::Session> current();

  Ort::Env env_;
  Ort::SessionOptions opts_;
  Ort::MemoryInfo mem_info_;

  std::mutex swap_mu_;                       // guards the pointer swap
  std::shared_ptr<Ort::Session> session_;    // current model

  int num_bins_ = 200;
};

}  // namespace bidder

#endif
