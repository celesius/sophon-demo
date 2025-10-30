#pragma once
// Clean public API for ArcFace — NO 3rd-party headers.
#include <cstdint>
#include <string>
#include <vector>

namespace arcface {

struct ImageView {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;
  int channels = 3;  // 1 or 3
};

struct Embedding {
  std::vector<float> data;  // L2-normalized feature
};

class Recognizer {
public:
  static Recognizer* create(const std::string& model_path, int device_id = 0);
  static void destroy(Recognizer*& ptr);

  Embedding embed(const ImageView& img);
  Embedding embed_from_file(const std::string& image_path);

  static float cosine(const Embedding& a, const Embedding& b);

private:
  Recognizer() = default;
  void* impl_ = nullptr;
};

} // namespace arcface
