#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


#include "bm_wrapper.hpp"
#include "bmnn_utils.h"
#include "cvwrapper.h"
#include "engine.h"
#include "utils.hpp"
#include "scrfd.hpp"

// InsightFace 112x112 五点模板
static const float ARC_SRC_5PTS_[5][2]
    = { { 38.2946f, 51.6963f }, { 73.5318f, 51.5014f }, { 56.0252f, 71.7366f }, { 41.5493f, 92.3655f }, { 70.7299f, 92.2041f } };

/*
struct cvai_pts_t {
    float* x;
    float* y;
    int size;  // 基本为5
};
*/

class ArcFaceBM {
 public:
    // model_path: ArcFace bmodel，比如 w600k_r50_f16_bm1688.bmodel
    // dev_id:     设备号
    // 推理引擎初始化
    explicit ArcFaceBM( int dev_id, const std::string& model_path, const std::string& gallery);
    //析构函数
    ~ArcFaceBM();
    // 类似scrfd的Init
    int Init(const float& sim_thresh=0.38);
    //数据前处理
    //1.bmimage转cv::mat
    //2. ...
    int pre_process(sail::BMImage& input);
    /**
     * @brief bm_image to cv::Mat
     * @param input  bm_image
     * @param output cv::Mat
     * @return 0 if success, -1 if fail
    */
    int bm_image_2_cv_image(sail::BMImage& input, cv::Mat& output);

    // 五点对齐 → 112x112（RGB）
    // pts：SCRFD 返回的五点（cvai_pts_t）
    static cv::Mat Align112By5Pts(const cv::Mat& bgr, const cvai_pts_t& pts);

    // 从 112x112 RGB 图像做前处理（[-1,1], NCHW）→ float
    static void Preprocess112RGB(const cv::Mat& rgb112, std::vector<float>& outCHW);

    // 单张前向，返回 512 维 L2 归一化特征（失败返回空向量）
    std::vector<float> Forward112RGB(const cv::Mat& rgb112);

    // 直接用五点（从整图裁剪对齐）
    std::vector<float> ForwardByKps(const cv::Mat& bgr, const cvai_pts_t& pts);

    // 余弦相似度
    static float CosineSim(const std::vector<float>& a, const std::vector<float>& b);

    // 输入张量尺寸 (1,3,112,112)
    inline int inC() const { return 3; }
    inline int inH() const { return 112; }
    inline int inW() const { return 112; }

 private:
#ifdef ARC_USE_WRAPPER
    // 你的工程里的自定义引擎包装（如果有）
    std::shared_ptr<Engine> engine_;  // 来自 engine.h
    std::string graph_name_, input_name_, output_name_;
#else
    // 直接用 SAIL
    std::shared_ptr<sail::Engine> engine_;
    std::shared_ptr<sail::Bmcv> bmcv_;
    //std::string graph_name_, input_name_, output_name_;
#endif
    float sim_thresh_ = 0.38f;  // 相似度阈值，默认0.38
    std::vector<int> input_shape_;  // [1,3,112,112]
    std::vector<std::string> graph_names_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int>> output_shape_;  // 9 outputs

    bm_data_type_t input_dtype_;
    bm_data_type_t output_dtype_;
    std::shared_ptr<sail::Tensor> input_tensor_;
    std::vector<std::shared_ptr<sail::Tensor>> output_tensor_;
    std::map<std::string, sail::Tensor*> input_tensors_;
    std::map<std::string, sail::Tensor*> output_tensors_;

    int m_net_h_, m_net_w_;
    int max_batch_;
    int min_dim_;
    float ab_[6];

};