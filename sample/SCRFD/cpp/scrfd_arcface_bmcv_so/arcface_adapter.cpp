#include "arcface.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Include your original implementation — private to the .so
#include "../third_party_orig/arcface.hpp"  // user's current header
#include "../third_party_orig/scrfd.hpp"    // if you need 5pts type
#include "bmnn_utils.h"
#include "bmruntime_interface.h"

namespace arcface {

namespace {

struct Impl {
  int device_id = 0;
  std::shared_ptr<BMNNContext> ctx;
  std::unique_ptr<ArcFaceBM> rec;  // user's recognizer

  bool init(const std::string& model_path, int dev) {
      device_id = dev;

      // 1) 先创建 BMNNHandle（通常 BMNNHandle(int dev_id) 构造）
      BMNNHandlePtr handle = std::make_shared<BMNNHandle>(device_id);
      if (!handle || !handle->handle()) {
          return false;
      }

      // 2) 用 (handle, bmodel_path) 构造 BMNNContext
      try {
          ctx = std::make_shared<BMNNContext>(handle, model_path.c_str());
      } catch (...) {
          return false;
      }

      // 3) 你的 ArcFace 实例
      rec = std::make_unique<ArcFaceBM>(ctx);
      if (!rec)
          return false;

      // 根据你的实现初始化识别器（如果需要阈值参数可从外层传）
      return rec->Init(/*sim_thresh*/ 0.38f) == 0;
  }

  Embedding run_from_view(const ImageView& iv) {
    Embedding e;
    // If your ArcFace supports RGB planar 112x112, convert as needed here
    // For demo simplicity, use a file-based path in Recognizer::embed_from_file
    (void)iv;
    return e;
  }

  Embedding run_from_file(const std::string& image_path) {
    Embedding e;
    // Typical pipeline in your demo:
    // 1) Decode to bm_image / cv::Mat
    // 2) Align face by 5pts (optional; if you already have aligned 112x112, skip)
    // 3) rec->Forward112BGR / Forward112RGB
    // Here we just show the concept:
    // cv::Mat bgr = cv::imread(image_path);
    // auto vec = rec->Forward112BGR(bgr);
    // e.data = std::move(vec);
    return e;
  }
};

} // namespace

Recognizer* Recognizer::create(const std::string& model_path, int device_id) {
  auto* p = new Recognizer();
  p->impl_ = new Impl();
  if (!static_cast<Impl*>(p->impl_)->init(model_path, device_id)) {
    delete static_cast<Impl*>(p->impl_);
    p->impl_ = nullptr;
    delete p;
    return nullptr;
  }
  return p;
}

void Recognizer::destroy(Recognizer*& ptr) {
  if (!ptr) return;
  delete static_cast<Impl*>(ptr->impl_);
  ptr->impl_ = nullptr;
  delete ptr;
  ptr = nullptr;
}

Embedding Recognizer::embed(const ImageView& img) {
  auto* impl = static_cast<Impl*>(impl_);
  if (!impl) return {};
  return impl->run_from_view(img);
}

Embedding Recognizer::embed_from_file(const std::string& image_path) {
  auto* impl = static_cast<Impl*>(impl_);
  if (!impl) return {};
  return impl->run_from_file(image_path);
}

float Recognizer::cosine(const Embedding& a, const Embedding& b) {
  if (a.data.empty() || b.data.empty() || a.data.size() != b.data.size()) return 0.f;
  double dot=0, na=0, nb=0;
  for (size_t i=0;i<a.data.size();++i) {
    dot += a.data[i]*b.data[i];
    na  += a.data[i]*a.data[i];
    nb  += b.data[i]*b.data[i];
  }
  na = std::sqrt(na)+1e-9; nb = std::sqrt(nb)+1e-9;
  return static_cast<float>(dot/(na*nb));
}

} // namespace arcface
