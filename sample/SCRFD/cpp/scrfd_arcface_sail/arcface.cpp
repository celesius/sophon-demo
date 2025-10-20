#include "arcface.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#ifndef ARC_USE_WRAPPER
//#include <sophon/sail/pytorch/sail_engine.hpp>  // 头文件路径按你的SDK调整
//using namespace sophon;
#endif

// ============== 辅助 ==============
static inline cv::Mat BGR2RGB(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}
static inline cv::Mat RGB2BGR(const cv::Mat& rgb) {
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}
static inline void L2Normalize(std::vector<float>& x) {
    double sum2 = 0.0;
    for (float v : x)
        sum2 += double(v) * double(v);
    if (sum2 <= 1e-12)
        return;
    float inv = 1.0f / float(std::sqrt(sum2));
    for (auto& v : x)
        v *= inv;
}

// ============== 对齐 ==============
cv::Mat ArcFaceBM::Align112By5Pts(const cv::Mat& bgr, const cvai_pts_t& pts) {
    if (pts.size < 5 || !pts.x || !pts.y)
        return cv::Mat();
    cv::Point2f src[5];
    for (int i = 0; i < 5; ++i)
        src[i] = cv::Point2f(pts.x[i], pts.y[i]);

    cv::Point2f dst[5];
    for (int i = 0; i < 5; ++i)
        dst[i] = cv::Point2f(ARC_SRC_5PTS_[i][0], ARC_SRC_5PTS_[i][1]);

    //cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::LMEDS).getMat(0);
    printf(" debug: %s %d \n", __func__, __LINE__);
    cv::Mat M;
    if (M.empty())
        return cv::Mat();

    cv::Mat rgb = BGR2RGB(bgr);
    cv::Mat aligned;
    cv::warpAffine(rgb, aligned, M, cv::Size(112, 112), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return aligned;  // RGB 112x112
}

// ============== 前处理 ==============
void ArcFaceBM::Preprocess112RGB(const cv::Mat& rgb112, std::vector<float>& outCHW) {
    CV_Assert(rgb112.data && rgb112.type() == CV_8UC3);
    cv::Mat f32;
    rgb112.convertTo(f32, CV_32FC3);
    f32 = f32 * (1.0 / 127.5) - 1.0;  // [-1,1]

    // HWC -> CHW
    outCHW.resize(3 * 112 * 112);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 112; ++y) {
            const float* row = f32.ptr<float>(y);
            for (int x = 0; x < 112; ++x) {
                outCHW[idx++] = row[x * 3 + c];
            }
        }
    }
}

// ============== 构造 ==============
ArcFaceBM::ArcFaceBM(int dev_id, const std::string& model_path, const std::string& gallery) : engine_() {
    //仿照scrfd的写法
    this->engine_ = std::make_shared<sail::Engine>(dev_id);   
    printf(" ArcFaceBM ctor .. dev_id=%d model=%s\n", dev_id, model_path.c_str());
    if (!this->engine_->load(model_path)) {
        std::cout << "ARCFACE Engine load bmodel " << model_path<< "failed" << std::endl;
        exit(0);
    }

    std::cout << "[ArcFace] model: " << model_path << "  input=" << input_shape_[0] << "x" << input_shape_[1] << "x" << input_shape_[2] << "x"
              << input_shape_[3] << std::endl;
}

ArcFaceBM::~ArcFaceBM() { std::cout << " ArcFaceBM ctor .." << std::endl; }

int ArcFaceBM::Init(const float& sim_thresh) {
    sim_thresh_ = sim_thresh;
    std::cout << "===============================" << std::endl;
    
    // 1. Initialize bmcv
    sail::Handle handle(engine_->get_device_id());
    bmcv_ = std::make_shared<sail::Bmcv>(handle);
   
    // 2. Initialize engine
    graph_names_ = engine_->get_graph_names();
    std::string gh_info;
    for_each(graph_names_.begin(), graph_names_.end(),
            [&](std::string& s) { gh_info += "0: " + s + "; "; });
    std::cout << "grapgh name -> " << gh_info << "\n";
    if (graph_names_.size() > 1) {
        std::cout << "NetworkNumError, this net only accept one network!"
                << std::endl;
        exit(1);
    }

    // input names of network
    input_names_ = engine_->get_input_names(graph_names_[0]);
    assert(input_names_.size() > 0);
    std::string input_tensor_names;
    for_each(input_names_.begin(), input_names_.end(),
            [&](std::string& s) { input_tensor_names += "0: " + s + "; "; });
    std::cout << "net input name -> " << input_tensor_names << "\n";
    if (input_names_.size() > 1) {
        std::cout << "InputNumError, Scrfd has only one inputs!" << std::endl;
        exit(1);
    }

    // output names of network
    output_names_ = engine_->get_output_names(graph_names_[0]);
    assert(output_names_.size() > 0);
    std::string output_tensor_names;
    for_each(output_names_.begin(), output_names_.end(),
            [&](std::string& s) { output_tensor_names += "0: " + s + "; "; });
    std::cout << "net output name -> " << output_tensor_names << "\n";

    // input shape of network 0
    input_shape_ = engine_->get_input_shape(graph_names_[0], input_names_[0]);
    std::string input_tensor_shape;
    for_each(input_shape_.begin(), input_shape_.end(),
            [&](int s) { input_tensor_shape += std::to_string(s) + " "; });
    std::cout << "input tensor shape -> " << input_tensor_shape << "\n";

    // output shapes of network 0
    output_shape_.resize(output_names_.size());
    for (int i = 0; i < output_names_.size(); i++) {
        output_shape_[i] = engine_->get_output_shape(graph_names_[0], output_names_[i]);
        std::string output_tensor_shape;
        for_each(output_shape_[i].begin(), output_shape_[i].end(),
                [&](int s) { output_tensor_shape += std::to_string(s) + " "; });
        std::cout << "output tensor " << i << " shape -> " << output_tensor_shape
                << "\n";
    }

    // data type of network input.
    input_dtype_ = engine_->get_input_dtype(graph_names_[0], input_names_[0]);
    std::cout << "input dtype -> " << input_dtype_
                << ", is fp32=" << ((input_dtype_ == BM_FLOAT32) ? "true" : "false")
                << "\n";

    // data type of network output.
    output_dtype_ = engine_->get_output_dtype(graph_names_[0], output_names_[0]);
    std::cout << "output dtype -> " << output_dtype_
                << ", is fp32=" << ((output_dtype_ == BM_FLOAT32) ? "true" : "false")
                << "\n";
    std::cout << "===============================" << std::endl;

    // 3. Initialize Network IO
    input_tensor_ = std::make_shared<sail::Tensor>(handle, input_shape_,
                                                    input_dtype_, false, false);
    input_tensors_[input_names_[0]] = input_tensor_.get();
    output_tensor_.resize(output_names_.size());
    for (int i = 0; i < output_names_.size(); i++) {
        output_tensor_[i] = std::make_shared<sail::Tensor>(handle, output_shape_[i],
                                                        output_dtype_, true, true);
        output_tensors_[output_names_[i]] = output_tensor_[i].get();
    }
    engine_->set_io_mode(graph_names_[0], sail::SYSO);

    // Initialize net utils 
    max_batch_ = input_shape_[0];
    m_net_h_ = input_shape_[2];
    m_net_w_ = input_shape_[3];

    min_dim_ = output_shape_[0].size();
    float input_scale = engine_->get_input_scale(graph_names_[0], input_names_[0]);
    input_scale = input_scale * 1.0 / 255.f;
    ab_[0] = 1.0 * input_scale;    // scale
    ab_[1] = 127.5 * input_scale;  //  mean
    ab_[2] = 1.0 * input_scale;
    ab_[3] = 127.5 * input_scale;
    ab_[4] = 1.0 * input_scale;
    ab_[5] = 127.5 * input_scale;
    std::cout << "input scale: " << input_scale << std::endl;
    
    return 0;
}


// ============== 推理 ==============
std::vector<float> ArcFaceBM::Forward112RGB(const cv::Mat& rgb112){
    if (rgb112.empty() || rgb112.rows != 112 || rgb112.cols != 112 || rgb112.channels() != 3)
        return {};

    std::vector<float> chw;
    Preprocess112RGB(rgb112, chw);

#ifdef ARC_USE_WRAPPER
    // —— 你的 Engine 包装写法（示例）——
    // inputs: map<input_name, float* 或 Tensor>
    // outputs: map<output_name, vector<float>>
    std::vector<float> feat;
    feat.resize(512);
    engine_->forward(input_name_, chw.data(), input_shape_, output_name_, feat.data(), { 1, 512 });
    L2Normalize(feat);
    return feat;
#else
    // —— SAIL —— SYSIO 模式直接给 host buffer
    std::map<std::string, std::vector<float>> inputs;
    //inputs[input_names_] = std::move(chw);
    auto bmimg = bmcv_->tensor_to_bm_image(*input_tensors_[input_names_[0]]);
    //auto outputs = engine_->process(graph_name_, inputs);
    engine_->process(graph_names_[0], input_tensors_, output_tensors_);
    //auto it = outputs.find(output_name_);
    //if (it == outputs.end())
    //    return {};

    // 拿到 1x512
    //const std::vector<float>& raw = it->second;
    std::vector<float> raw;
    std::vector<float> feat = raw;
    if (feat.size() != 512) {
        // 某些模型输出 (1,512) 也可能是 512 长度；如果是 (N,512) 则需要自己 reshape 后取第0条
        // 这里简单兜底
        if (feat.size() > 512)
            feat.resize(512);
    }
    L2Normalize(feat);
    return feat;
#endif
}

std::vector<float> ArcFaceBM::ForwardByKps(const cv::Mat& bgr, const cvai_pts_t& pts){
    cv::Mat rgb112 = Align112By5Pts(bgr, pts);
    if (rgb112.empty())
        return {};
    return Forward112RGB(rgb112);
}

float ArcFaceBM::CosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty())
        return -1.f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 1e-12 || nb <= 1e-12)
        return -1.f;
    return float(dot / std::sqrt(na * nb));
}