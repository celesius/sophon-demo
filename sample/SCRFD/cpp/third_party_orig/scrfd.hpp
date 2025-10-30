#pragma once
// Clean public API for SCRFD — NO 3rd-party headers.
// Depends only on the C++ standard library.
#include <cstdint>
#include <string>
#include <vector>

namespace scrfd {

struct FaceBox {
  float x1, y1, x2, y2;
  float score;
};

struct Landmark5 {
  float x[5];
  float y[5];
};

struct Face {
  FaceBox box;
  Landmark5 lmk;
};

struct Face_3D_Landmark{
    float x, y, z;
};

struct Faces_Landmarks {
    std::vector<Face> faces;
    std::vector<Face_3D_Landmark> landmarks3d;
    /* data */
};

struct ImageView {
  // Interleaved RGB or Gray buffer.
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;    // bytes per row; if 0, treated as width*channels
  int channels = 3;  // 1 or 3
};

class Detector {
public:
  // Create a detector with model path and optional device id (TPU index).
  static Detector* create(const std::string& model_path, int device_id = 0);
  static Detector* create_with_arcface(const std::string& model_path, const std::string& arcface_model_path, const std::string& face3d_model_path, int device_id = 0,
                          const std::string& gallery_path = "",
                          const std::string& meanshape_path = "",
                          float score_thresh = 0.5f,
                          float nms_iou = 0.4f,
                          float sim_thresh = 0.38f
  );

  

  
  static void destroy(Detector*& ptr);

  // Detect faces from an in-memory image view.
  //std::vector<Face> detect(const ImageView& img,
  Faces_Landmarks detect(const ImageView& img,
                          float score_thresh = 0.5f,
                          float nms_iou = 0.4f
                        );
  
  std::vector<Face> detect_with_arcface(const ImageView& img);



  // Convenience: detect from an image file path.
  std::vector<Face> detect_from_file(const std::string& image_path,
                                     float score_thresh = 0.5f,
                                     float nms_iou = 0.4f);

private:
  Detector() = default;
  void* impl_ = nullptr; // PIMPL: implementation hidden inside the .so
};

} // namespace scrfd
