#include "scrfd.hpp"
#include "arcface.hpp"
#include "face3d.hpp"

// Adapter that bridges to the user's existing implementation in scrfd.hpp/scrfd.cpp
// This .cpp can include any third-party headers you need; they will NOT leak to the public API.
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// === Include your original headers (private to the .so) ===
#include "../third_party_orig/scrfd.hpp"     // <-- your current working header (includes bmnn_utils, opencv, etc.)
#include "bmnn_utils.h"                      // SDK private headers (example)
#include "bmruntime_interface.h"
#include "ff_decode.hpp"
//#include "opencv2/opencv.hpp"

using namespace face3d;

namespace scrfd {

namespace {

    /*
    cv::Mat imageViewToMat(const scrfd::ImageView& iv) {
        int type = (iv.channels == 3) ? CV_8UC3 : (iv.channels == 1) ? CV_8UC1 : -1;
        if (type < 0) {
            throw std::runtime_error("Unsupported channel count in ImageView");
        }

        // stride 是每行字节数（不一定等于 width * channels）
        cv::Mat mat(iv.height, iv.width, type, const_cast<uint8_t*>(iv.data), iv.stride);
        return mat.clone();  // clone to ensure data ownership
    }
        */

    static void pack_rows(const uint8_t* src, int h, int row_bytes, int stride, std::vector<uint8_t>& tight) {
        tight.resize(static_cast<size_t>(h) * row_bytes);
        if (stride == row_bytes) {
            std::memcpy(tight.data(), src, tight.size());
        } else {
            for (int y = 0; y < h; ++y) {
                std::memcpy(tight.data() + static_cast<size_t>(y) * row_bytes, src + static_cast<size_t>(y) * stride, row_bytes);
            }
        }
    }

    // 成功返回 true；失败会销毁 out_img 并返回 false。
    static bool imageview_to_bm_image_rgb_packed(bm_handle_t handle, const uint8_t* data, int w, int h, int stride, int channels, bm_image& out_img) {
        if (!data || w <= 0 || h <= 0 || (channels != 1 && channels != 3))
            return false;

        // 1) 将输入整理为 RGB 打包：GRAY 扩展成 RGB，RGB 直接按行打包
        std::vector<uint8_t> host_rgb;
        host_rgb.reserve(static_cast<size_t>(w) * h * 3);

        if (channels == 3) {
            // 输入已经是 RGB
            pack_rows(data, h, w * 3, (stride > 0 ? stride : w * 3), host_rgb);
        } else {  // channels == 1
            // GRAY → RGB
            const int in_stride = (stride > 0 ? stride : w);
            host_rgb.resize(static_cast<size_t>(w) * h * 3);
            for (int y = 0; y < h; ++y) {
                const uint8_t* src_row = data + static_cast<size_t>(y) * in_stride;
                uint8_t* dst_row = host_rgb.data() + static_cast<size_t>(y) * w * 3;
                for (int x = 0; x < w; ++x) {
                    uint8_t v = src_row[x];
                    dst_row[3 * x + 0] = v;
                    dst_row[3 * x + 1] = v;
                    dst_row[3 * x + 2] = v;
                }
            }
        }

        // 2) 创建 device 侧 bm_image（RGB 打包，u8）
        bm_status_t ret = bm_image_create(handle, h, w, FORMAT_RGB_PACKED, DATA_TYPE_EXT_1N_BYTE, &out_img);
        if (ret != BM_SUCCESS)
            return false;

        // 3) 分配 device 内存（关键！）
        ret = bm_image_alloc_dev_mem(out_img, BMCV_HEAP_ANY);
        if (ret != BM_SUCCESS) {
            bm_image_destroy(out_img);
            return false;
        }

        // 4) Host → Device 拷贝
        void* planes[1] = { host_rgb.data() };
        ret = bm_image_copy_host_to_device(out_img, planes);
        if (ret != BM_SUCCESS) {
            bm_image_destroy(out_img);
            return false;
        }

        return true;
    }

// Convert our clean Face to the original types if needed (here we fill from original).
inline Face make_face(cvai_face_info_t& src) {
  Face f{};
  f.box.x1 = src.bbox.x1;
  f.box.y1 = src.bbox.y1;
  f.box.x2 = src.bbox.x2;
  f.box.y2 = src.bbox.y2;
  f.box.score = src.bbox.score;
  for (int i=0;i<5;++i) {
    f.lmk.x[i] = src.pts.x[i];
    f.lmk.y[i] = src.pts.y[i];
  }
  return f;
}

// 兜底：把 src 放到 target_handle 所属的 bm_image 里（同尺寸/同格式/同数据类型）
static bool ensure_image_on_handle(bm_image& src, bm_handle_t target_handle, bm_image& dst_same_handle) {
    // 取 meta
    int w = 0, h = 0;
    bm_image_format_info fmt;
    bm_image_data_format_ext dtype;
    bm_handle_t src_handle = nullptr;

    src_handle = bm_image_get_handle(&src);
    bm_image_get_format_info(&src, &fmt);
    dtype = src.data_type;
    w = src.width;
    h = src.height;
    

    // 已在同一 handle，零拷贝直通（浅拷贝）
    if (src_handle == target_handle) {
        dst_same_handle = src;
        return true;
    }

    // 否则分配一张同规格的 dst，并 Host 中转复制
    if (bm_image_create(target_handle, h, w, fmt.image_format, dtype, &dst_same_handle) != BM_SUCCESS)
        return false;
    if (bm_image_alloc_dev_mem(dst_same_handle, BMCV_HEAP1_ID) != BM_SUCCESS)
        return false;

    //size_t bytes[4] = { 0 };
    int bytes = 0;
    bm_image_get_byte_size(src, &bytes);
    //std::vector<uint8_t> host(bytes[0]);
    uint8_t* host= new uint8_t[bytes];
    memset(host, 0, bytes);
    // D2H
    if (bm_image_copy_device_to_host(src, (void **)&host) != BM_SUCCESS){
        printf("bm_image_copy_device_to_host failed\n");
        delete[] host;
        return false;
    }
    // H2D
    if (bm_image_copy_host_to_device(dst_same_handle, (void **)&host) != BM_SUCCESS){
        printf("bm_image_copy_host_to_device failed\n");
        delete[] host;
        return false;
    }
    printf("ensure_image_on_handle success\n");
    delete[] host;
    return true;
}

struct Impl {
  int device_id = 0;
  BMNNHandlePtr handle;
  std::shared_ptr<BMNNContext> bm_ctx;
  std::shared_ptr<BMNNContext> arc_ctx;
  std::shared_ptr<BMNNContext> face3d_ctx;
  std::unique_ptr<ArcFaceBM> arcface;
  std::unique_ptr<Scrfd> det;   // user's original detector
  std::unique_ptr<Face3D> face3d;
  TimeStamp scrfd_ts;
  bm_image in_dev_ {};
  bool in_dev_ok_ = false;
  int in_w_ = 0, in_h_ = 0;

  ~Impl() {
      if (in_dev_ok_){
          bm_image_destroy(in_dev_);
      }
  }

  bool init(const std::string& model_path, int dev) {
      device_id = dev;
      handle = std::make_shared<BMNNHandle>(device_id);
      std::cout << "set device id: " << dev << std::endl;
      bm_handle_t h = handle->handle();

      try {
          bm_ctx = std::make_shared<BMNNContext>(handle, model_path.c_str());
      } catch (...) {
          return false;
      }

        det = std::make_unique<Scrfd>(bm_ctx);
      if (!det)
          return false;
      CV_Assert(0 == det->Init(0.5,
                            0.4));
      // profiling
      //TimeStamp* ts = &scrfd_ts;
      det->enableProfile(&scrfd_ts);

      printf("1 det pointer: %p\n", static_cast<void*>(det.get()));
      // get batch_size
      int batch_size = det->batch_size();
      printf("Scrfd initialized on device %d with model %s (batch_size=%d)\n",
             device_id, model_path.c_str(), batch_size);

      return true;
  }

  bool init_with_arcface(const std::string& model_path, 
                         const std::string& arcface_path, 
                         const std::string& face3d_path,
                         int dev,
                          const std::string& gallery_path,
                          const std::string& meanshape_path,
                          float score_thresh = 0.5f,
                          float nms_iou = 0.4f,
                          float sim_thresh = 0.38f) 
                          {
      device_id = dev;
      handle = std::make_shared<BMNNHandle>(device_id);
      std::cout << "init_with_arcface set device id: " << dev << std::endl;
      //bm_handle_t h = handle->handle();

      try {
            bm_ctx = std::make_shared<BMNNContext>(handle, model_path.c_str());
            arc_ctx= std::make_shared<BMNNContext>(handle, arcface_path.c_str());
            face3d_ctx = std::make_shared<BMNNContext>(handle, face3d_path.c_str());
        } catch (...) {
          return false;
      }

        det = std::make_unique<Scrfd>(bm_ctx);
        arcface = std::make_unique<ArcFaceBM>(arc_ctx);
        face3d = std::make_unique<Face3D>(face3d_ctx);
      if (!det || !arcface || !face3d){
          return false;
      }
        CV_Assert(0 == det->Init(score_thresh ,nms_iou));
        CV_Assert(0 == arcface->Init(sim_thresh));
        Face3D::Config face3d_cfg;
        face3d_cfg.meanshape_txt = meanshape_path;
        printf("meanshape path: %s\n", face3d_cfg.meanshape_txt.c_str());
        CV_Assert(0 == face3d->Init(face3d_cfg));
        printf("face3d initialized\n");
      // profiling
      //TimeStamp* ts = &scrfd_ts;
      det->enableProfile(&scrfd_ts);

      printf("1 det pointer: %p\n", static_cast<void*>(det.get()));
      // get batch_size
      int batch_size = det->batch_size();
      printf("Scrfd initialized on device %d with model %s (batch_size=%d)\n",
             device_id, model_path.c_str(), batch_size);

      return true;
  }


  std::vector<Face> run_from_file(const std::string& image_path, float score, float nms){
    // Prepare bm_image and call user's Scrfd::Detect
    // This path uses user's helper to decode into bm_image (preferred in demos).
    std::vector<bm_image> batch_imgs;
    bm_handle_t bm_h = handle->handle();  // bm_ctx->handle(); // typical API in bmnn_utils
    bm_image bmimg;
    // NOTE: You'll likely have a helper like picDec(bm_h, image_path.c_str(), bmimg)
    // Here we assume such a function exists; otherwise, replace with your actual decoder.
    picDec(bm_h, image_path.c_str(), bmimg);
    batch_imgs.push_back(bmimg);

    std::vector<ScrfdBoxVec> batch_boxes;
    int ret = det->Detect(batch_imgs, batch_boxes);

    // Cleanup input images
    for (auto& im : batch_imgs) bm_image_destroy(im);
    if (ret != 0 || batch_boxes.empty()) return {};
    // Convert back to clean Face
    std::vector<Face> faces;
    for (auto& fi : batch_boxes[0]) {
      // ScrfdBox (fi) -> Face; many demos pack into cvai_face_info_t
      cvai_face_info_t cvf{};
      cvf.bbox = fi.bbox;
      cvf.pts = fi.pts;
      // Landmarks fill if you have them (demo dependent). Here we skip if unavailable.
      faces.push_back(make_face(cvf));
    }
    return faces;
  }

  /*
  std::vector<Face> run_from_view(const ImageView& iv, float score, float nms){
    // If your pipeline accepts bm_image only, convert from CPU buffer -> bm_image -> Detect
    // TODO: Implement bm_image_from_buffer + bm_image_copy_host_to_device in your project style.
    //(void)iv; (void)score; (void)nms;
    std::vector<Face> faces;

    std::vector<bm_image> input_images;
    bm_image _bmimg;
    if (!imageview_to_bm_image_rgb_packed(bm_ctx->handle(), iv.data, iv.width, iv.height, (iv.stride > 0 ? iv.stride : iv.width * iv.channels), iv.channels,
                                          _bmimg)) {
        return faces;
    }

    //cv::bmcv::toBMI(rgb_mat, &_bmimg);
    input_images.push_back(_bmimg);
    std::vector<ScrfdBoxVec> batch_boxes;
    CV_Assert(0 == det->Detect(input_images, batch_boxes));
    
    for (auto& fi : batch_boxes[0]) {
      // ScrfdBox (fi) -> Face; many demos pack into cvai_face_info_t
      cvai_face_info_t cvf{};
      cvf.bbox = fi.bbox;
      cvf.pts = fi.pts;
      // Landmarks fill if you have them (demo dependent). Here we skip if unavailable.
      faces.push_back(make_face(cvf));
    }

    return faces;
  }
  */
  std::vector<Face> run_from_view(const ImageView& iv, float score, float nms, Landmarks3DOut& lmk_out) {
      std::vector<Face> faces;
      if (!iv.data || iv.width <= 0 || iv.height <= 0)
          return faces;

      // 1) 如果尺寸变了，重建 device 图像
      if (!in_dev_ok_ || iv.width != in_w_ || iv.height != in_h_) {
          if (in_dev_ok_) {
              bm_image_destroy(in_dev_);
              in_dev_ok_ = false;
          }
          // 创建 + 分配（RGB_PACKED/u8）
          if (BM_SUCCESS != bm_image_create(bm_ctx->handle(), iv.height, iv.width, FORMAT_RGB_PACKED, DATA_TYPE_EXT_1N_BYTE, &in_dev_)) {
              return faces;
          }
          if (BM_SUCCESS != bm_image_alloc_dev_mem(in_dev_, BMCV_HEAP_ANY)) {
              bm_image_destroy(in_dev_);
              return faces;
          }
          in_w_ = iv.width;
          in_h_ = iv.height;
          in_dev_ok_ = true;
      }

      // 2) 准备 host RGB（将灰度扩展成RGB；如果你外层就是RGB，这步可省）
      std::vector<uint8_t> host_rgb;
      const int in_stride = (iv.stride > 0 ? iv.stride : iv.width * iv.channels);
      if (iv.channels == 3) {
          // 打包行
          host_rgb.resize((size_t)iv.width * iv.height * 3);
          for (int y = 0; y < iv.height; ++y) {
              std::memcpy(host_rgb.data() + (size_t)y * iv.width * 3, iv.data + (size_t)y * in_stride, iv.width * 3);
          }
      } else {
          // 灰度拓展成 RGB
          host_rgb.resize((size_t)iv.width * iv.height * 3);
          for (int y = 0; y < iv.height; ++y) {
              const uint8_t* src = iv.data + (size_t)y * in_stride;
              uint8_t* dst = host_rgb.data() + (size_t)y * iv.width * 3;
              for (int x = 0; x < iv.width; ++x) {
                  uint8_t v = src[x];
                  dst[3 * x + 0] = v;
                  dst[3 * x + 1] = v;
                  dst[3 * x + 2] = v;
              }
          }
      }

      // 3) Host → Device（复用已分配的 device mem）
      void* planes[1] = { host_rgb.data() };
      if (BM_SUCCESS != bm_image_copy_host_to_device(in_dev_, planes)) {
          return faces;
      }

      // 4) Detect
      //det->Init(score, nms);
      std::vector<bm_image> input_images { in_dev_ };
      std::vector<ScrfdBoxVec> batch_boxes;
      int ret = det->Detect(input_images, batch_boxes);
      if (ret != 0 || batch_boxes.empty())
          return faces;

      // 5) 结果
      faces.reserve(batch_boxes[0].size());
      for (auto& fi : batch_boxes[0]) {
          cvai_face_info_t cvf {};
          cvf.bbox = fi.bbox;
          cvf.pts = fi.pts;
          faces.push_back(make_face(cvf));
      }

      std::vector<face3d::FaceBox> face_boxes;
        for (auto& f : faces) {
            face3d::FaceBox box;
            box.x1 = f.box.x1;
            box.y1 = f.box.y1;
            box.x2 = f.box.x2;
            box.y2 = f.box.y2;
            box.score = f.box.score;
            face_boxes.push_back(box);
        }

        face3d::Pose3DOut pose_out;
        bm_image face3d_image;
        
        if(ensure_image_on_handle(input_images[0], face3d_ctx->handle(), face3d_image)){
            printf("ensure_image_on_handle success\n");
        }else{
            printf("ensure_image_on_handle failed\n");
        }

        //Landmarks3DOut lmk_out; 

        bool ret_3d = face3d->infer(face3d_image, face_boxes, pose_out, lmk_out);
        if(ret_3d){
            printf("Pitch: %.2f, Yaw: %.2f, Roll: %.2f, s: %.4f\n",
                   pose_out.pitch_deg, pose_out.yaw_deg, pose_out.roll_deg, pose_out.s);
        }else{
            printf("Face3D infer failed!\n");
        }

      return faces;
  }
};

} // namespace

Detector* Detector::create(const std::string& model_path, int device_id) {
  auto* p = new Detector();
  p->impl_ = new Impl();
  if (!static_cast<Impl*>(p->impl_)->init(model_path, device_id)) {
    printf("Failed to initialize Scrfd detector with model %s\n", model_path.c_str());
    delete static_cast<Impl*>(p->impl_);
    p->impl_ = nullptr;
    delete p;
    return nullptr;
  }
  return p;
}

Detector* Detector::create_with_arcface(const std::string& model_path, const std::string& arcface_model_path, const std::string& face3d_model_path, int device_id,
                          const std::string& gallery_path,
                          const std::string& meanshape_path,
                          float score_thresh,
                          float nms_iou,
                          float sim_thresh
  ){
    printf("Creating Detector with SCRFD model: %s, ArcFace model: %s, Face3D model: %s\n", 
        model_path.c_str(), arcface_model_path.c_str(), face3d_model_path.c_str());
  
    auto* p = new Detector();
  p->impl_ = new Impl();
  //if (!static_cast<Impl*>(p->impl_)->init(model_path, device_id)) {
  if (!static_cast<Impl*>(p->impl_)->init_with_arcface(model_path, 
    arcface_model_path,
    face3d_model_path,
    device_id,
    gallery_path,
    meanshape_path,
    score_thresh,
    nms_iou,
    sim_thresh)) {
    
    //printf("Failed to initialize Scrfd detector with model %s\n", model_path.c_str());
    delete static_cast<Impl*>(p->impl_);
    p->impl_ = nullptr;
    delete p;
    return nullptr;
  }
  return p;

  }


void Detector::destroy(Detector*& ptr) {
  if (!ptr) return;
  delete static_cast<Impl*>(ptr->impl_);
  ptr->impl_ = nullptr;
  delete ptr;
  ptr = nullptr;
}

Faces_Landmarks Detector::detect(const ImageView& img, float score_thresh, float nms_iou) {
  auto* impl = static_cast<Impl*>(impl_);
  if (!impl) return {};
    Landmarks3DOut lmk_out;
  
  std::vector<Face> faces = impl->run_from_view(img, score_thresh, nms_iou, lmk_out);
    Faces_Landmarks result;
    result.faces = faces;
    std::vector<Face_3D_Landmark> lmk3d_vec;
    for (const auto& lmk : lmk_out.pts3d) {
        Face_3D_Landmark lmk3d;
        lmk3d.x = lmk.x;
        lmk3d.y = lmk.y;
        lmk3d.z = lmk.z;
        lmk3d_vec.push_back(lmk3d);
    }
    result.landmarks3d = lmk3d_vec;
  
  return result;
}

std::vector<Face> Detector::detect_from_file(const std::string& image_path, float score_thresh, float nms_iou) {
  auto* impl = static_cast<Impl*>(impl_);
  if (!impl) return {};
  return impl->run_from_file(image_path, score_thresh, nms_iou);
}

} // namespace scrfd
