#include "model_session.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace bidder {

ModelSession::ModelSession(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "bid_model"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  // the model is tiny, so let our worker threads do the parallelism, not ORT
  opts_.SetIntraOpNumThreads(1);
  opts_.SetInterOpNumThreads(1);
  opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = build(model_path);

  // figure out the number of bins from the model's output shape, so we don't
  // have to hardcode it anywhere else
  Ort::TypeInfo ti = session_->GetOutputTypeInfo(0);
  auto shape = ti.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() == 2 && shape[1] > 0) {
    num_bins_ = static_cast<int>(shape[1]);
  }
}

std::shared_ptr<Ort::Session> ModelSession::build(const std::string& model_path) {
  return std::make_shared<Ort::Session>(env_, model_path.c_str(), opts_);
}

std::shared_ptr<Ort::Session> ModelSession::current() {
  std::lock_guard<std::mutex> lk(swap_mu_);
  return session_;  // copy the shared_ptr so the session stays alive while we use it
}

void ModelSession::reload(const std::string& model_path) {
  auto fresh = build(model_path);

  // check the new model before trusting it: run one all-zeros input and make
  // sure the output is the right shape with no NaNs in it
  std::array<int64_t, 9> cat{};
  std::array<float, 9> cont{};
  std::array<int64_t, 10> tags{};
  std::array<int64_t, 2> cat_shape{1, 9};
  std::array<int64_t, 2> cont_shape{1, 9};
  std::array<int64_t, 2> tags_shape{1, 10};

  Ort::Value inputs[3] = {
      Ort::Value::CreateTensor<int64_t>(mem_info_, cat.data(), cat.size(),
                                        cat_shape.data(), cat_shape.size()),
      Ort::Value::CreateTensor<float>(mem_info_, cont.data(), cont.size(),
                                      cont_shape.data(), cont_shape.size()),
      Ort::Value::CreateTensor<int64_t>(mem_info_, tags.data(), tags.size(),
                                        tags_shape.data(), tags_shape.size()),
  };
  const char* in_names[] = {"cat", "cont", "tags"};
  const char* out_names[] = {"probs"};
  auto out = fresh->Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 1);
  auto info = out[0].GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 2 || shape[1] != num_bins_) {
    throw std::runtime_error("reload rejected: unexpected output shape");
  }
  const float* p = out[0].GetTensorData<float>();
  for (int i = 0; i < num_bins_; i++) {
    if (std::isnan(p[i]) || std::isinf(p[i])) {
      throw std::runtime_error("reload rejected: NaN/Inf in output");
    }
  }

  std::lock_guard<std::mutex> lk(swap_mu_);
  session_ = fresh;  // to a reader grabbing the pointer, this looks instant
}

void ModelSession::run(const int64_t* cat, const float* cont, const int64_t* tags,
                       int batch, std::vector<float>& out_probs) {
  auto sess = current();

  std::array<int64_t, 2> cat_shape{batch, 9};
  std::array<int64_t, 2> cont_shape{batch, 9};
  std::array<int64_t, 2> tags_shape{batch, 10};

  Ort::Value inputs[3] = {
      Ort::Value::CreateTensor<int64_t>(mem_info_, const_cast<int64_t*>(cat),
                                        static_cast<size_t>(batch) * 9,
                                        cat_shape.data(), cat_shape.size()),
      Ort::Value::CreateTensor<float>(mem_info_, const_cast<float*>(cont),
                                      static_cast<size_t>(batch) * 9,
                                      cont_shape.data(), cont_shape.size()),
      Ort::Value::CreateTensor<int64_t>(mem_info_, const_cast<int64_t*>(tags),
                                        static_cast<size_t>(batch) * 10,
                                        tags_shape.data(), tags_shape.size()),
  };
  const char* in_names[] = {"cat", "cont", "tags"};
  const char* out_names[] = {"probs"};

  auto out = sess->Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 1);
  const float* p = out[0].GetTensorData<float>();
  out_probs.resize(static_cast<size_t>(batch) * num_bins_);
  for (size_t i = 0; i < out_probs.size(); i++) {
    out_probs[i] = p[i];
  }
}

}  // namespace bidder
