#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "bm_wrapper.hpp"
#include "bmruntime_interface.h"
#include "bmnn_utils.h"
#include "utils.hpp"
#include "scrfd.hpp"
// 来自 SCRFD 的 5点结构
/*
typedef struct cvai_pts_t {
    float* x;
    float* y;
    int size;
} cvai_pts_t;
*/
class ArcFaceBM {
 public:
    explicit ArcFaceBM(std::shared_ptr<BMNNContext> ctx);
    ~ArcFaceBM();

    // 与 SCRFD 一致：Init 里完成网络、bm_image 申请、converto 参数设置
    int Init(float sim_thresh = 0.38f);

    // 5点对齐（直接对整图做仿射，不需要先 crop）
    static cv::Mat AlignBy5PtsRGB(const cv::Mat& bgr, const cvai_pts_t& pts, int out_w = 112, int out_h = 112);

    // 单张 112x112 RGB 前向（内部会走 bmcv_image_convert_to 到 [-1,1]、绑定 tensor、forward）
    std::vector<float> Forward112RGB(cv::Mat& rgb112);

    // BGR 输入也可（内部转换成 RGB 再走前向）
    std::vector<float> Forward112BGR(const cv::Mat& bgr112);

    // 余弦相似度
    static float CosineSim(const std::vector<float>& a, const std::vector<float>& b);

    // 读入图库（人名=子目录名），对每张图做检测外的人脸对齐+特征；你如果已有 Python 侧图库，就不用这个
    // 这里只留接口，便于后续扩展
    struct GalleryItem {
        std::string name;
        std::vector<float> feat;
    };
    std::vector<GalleryItem> gallery_;

    // 构建人脸图库（会清空旧数据）
    bool BuildGallery(const std::string& root, bm_handle_t& bm_h, Scrfd& detector, float det_conf);

    // 识别单张人脸特征与图库中最相似项
    bool MatchGallery(const std::vector<float>& feat, std::string& best_name, float& best_sim);

    int pre_process(const std::vector<bm_image>& input);
    
    float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                                bool* pIsAligWidth);
 
private:
    // 上传 + 预处理（把 HWC RGB 数据按通道拆成 plane，拷到 m_planar_imgs[0]，再用 convert_to → m_converto_imgs[0]）
    bool UploadAndPreprocess(cv::Mat& rgb112, bm_image& out_converto);

    // 只做一次推理 + 拷回 CPU
    //std::vector<float> InferenceOnce(const bm_image& input_converto);
    std::vector<float> InferenceOnce();

 private:
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;

    int m_net_h = 112;
    int m_net_w = 112;
    int max_batch = 1;

    float m_sim_thresh = 0.38f;

    // BMCV 资源（和 scrfd 一样风格）
    std::vector<bm_image> m_planar_imgs;    // RGB planar, uint8
    std::vector<bm_image> m_converto_imgs;  // after convert_to, float32/int8
    bmcv_convert_to_attr converto_attr {};  // alpha/beta

    // 为了使用 bm_image_get_contiguous_device_mem
    int aligned_w_ = 0;
};