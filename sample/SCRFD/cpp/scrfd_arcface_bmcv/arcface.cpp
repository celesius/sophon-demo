#include "arcface.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <numeric>  // for inner_product
#include <opencv2/opencv.hpp>
#include "ff_decode.hpp"

#include <regex>
#include <string>

namespace fs = std::filesystem;

// 自动生成下一个 dump 文件名
// base_dir: 保存目录（如 "debug"）
// prefix: 文件前缀（如 "dbg_arcface_planar"）
// suffix: 文件后缀（如 ".bmimg"）
std::string GetNextDumpFilename(const std::string& base_dir, const std::string& prefix, const std::string& suffix = ".jpg") {
    if (!fs::exists(base_dir)) {
        fs::create_directories(base_dir);
    }

    std::regex pattern(prefix + "_(\\d{4})" + suffix);
    int max_index = 0;

    for (auto& entry : fs::directory_iterator(base_dir)) {
        if (!entry.is_regular_file())
            continue;
        std::string name = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(name, match, pattern)) {
            int idx = std::stoi(match[1]);
            if (idx > max_index)
                max_index = idx;
        }
    }

    int next_index = max_index + 1;
    char buf[32];
    snprintf(buf, sizeof(buf), "_%04d", next_index);
    std::string filename = prefix + buf + suffix;

    return (fs::path(base_dir) / filename).string();
}

/*
static void DebugCheckBmImageF32RGBPlanar(bm_handle_t h, const bm_image& img, const char* tag = "converto", int sample_hw = 16) {
    // 简要信息
    int stride[4];
    bm_image_get_stride(img, stride);
    
    std::cout << "[DBG] " << tag << ": " << img.width << "x" << img.height << " fmt=" << img.image_format  // 期望 FORMAT_RGB_PLANAR
              << " dtype=" << img.data_type                                                                // 期望 DATA_TYPE_EXT_FLOAT32
              << " stride0=" << stride[0] << "\n";

    const int H = img.height, W = img.width;
    const int h_chk = std::min(sample_hw, H);
    const int w_chk = std::min(sample_hw, W);

    // 逐 plane 抽样拷回
    for (int c = 0; c < 3; ++c) {
        bm_device_mem_t dev;
        bm_image_get_device_mem(img, &dev);
        std::vector<float> buf(H * W);
        bm_memcpy_d2s(h, buf.data(), dev);  // device->system

        // 做个 16x16 的 min/max/NaN 检查
        bool has_nan = false, has_inf = false;
        float mn = +1e30f, mx = -1e30f;
        for (int y = 0; y < h_chk; ++y) {
            const float* row = buf.data() + y * W;
            for (int x = 0; x < w_chk; ++x) {
                float v = row[x];
                if (!std::isfinite(v)) {
                    has_nan |= std::isnan(v);
                    has_inf |= std::isinf(v);
                } else {
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                }
            }
        }
        std::cout << "   ch" << c << " sample[0:" << h_chk << ",0:" << w_chk << "] min=" << mn << " max=" << mx << " NaN=" << (has_nan ? "YES" : "no")
                  << " Inf=" << (has_inf ? "YES" : "no") << "\n";

        // 打印前 16 个值瞅一眼
        std::cout << "   ch" << c << " first16:";
        for (int i = 0; i < std::min(16, W * H); ++i)
            std::cout << " " << buf[i];
        std::cout << "\n";
    }
}
*/

static void DebugCheckBmImageF32RGBPlanar(bm_handle_t h, const bm_image& img, const char* tag = "converto", int sample_hw = 16) {
    int stride[4];
    bm_image_get_stride(img, stride);
    
    std::cout << "[DBG] " << tag << ": " << img.width << "x" << img.height << " fmt=" << img.image_format  // 期望 FORMAT_RGB_PLANAR
              << " dtype=" << img.data_type                                                                // 期望 DATA_TYPE_EXT_FLOAT32
              << " stride0=" << stride[0] << "\n";

    const int H = img.height, W = img.width;
    const int hw = std::min(sample_hw, H);
    const int ww = std::min(sample_hw, W);

    // 读取整张显存
    bm_device_mem_t dev;
    bm_image_get_device_mem(img, &dev);

    // 分配 host buffer 接收
    size_t byte_size = W * H * 3 * sizeof(float);
    std::vector<float> buf(W * H * 3);
    auto ret = bm_memcpy_d2s_partial(h, buf.data(), dev, byte_size);
    if (ret != BM_SUCCESS) {
        std::cerr << "[ERR] bm_memcpy_d2s_partial failed, ret=" << ret << std::endl;
        return;
    }

    // 做简单统计
    bool has_nan = false, has_inf = false;
    float mn = +1e30f, mx = -1e30f;
    for (int i = 0; i < hw * ww * 3; ++i) {
        float v = buf[i];
        if (!std::isfinite(v)) {
            has_nan |= std::isnan(v);
            has_inf |= std::isinf(v);
        } else {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
    }
    std::cout << "   sample [" << hw << "x" << ww << "] min=" << mn << " max=" << mx << " NaN=" << (has_nan ? "YES" : "no")
              << " Inf=" << (has_inf ? "YES" : "no") << std::endl;

    // 打印前 16 个值看分布
    std::cout << "   first16:";
    for (int i = 0; i < std::min(16, (int)buf.size()); ++i)
        std::cout << " " << buf[i];
    std::cout << std::endl;

    /*
    std::cout << " dump all input data to :" << std::endl;
    for (int c = 0; c < 3; ++c) {
        std::cout << "  channel " << c << ":" << std::endl;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                std::cout << buf[c * W * H + y * W + x] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    */
}

static void DebugCheckTensorInOut(BMNNTensor* in, BMNNTensor* out) {
    // 输入张量
    if (in) {
        const bm_shape_t* s = in->get_shape();
        std::cout << "[DBG] input tensor: dims=";
        for (int i = 0; i < s->num_dims; ++i)
            std::cout << s->dims[i] << (i + 1 < s->num_dims ? "x" : "");
        std::cout << " dtype=" << in->get_dtype() << " scale=" << in->get_scale() << "\n";
    }

    // 输出张量
    if (out) {
        // ✅ 删除 out->sync_d2s();
        const bm_shape_t* s = out->get_shape();
        int n = 1;
        for (int i = 0; i < s->num_dims; ++i)
            n *= s->dims[i];

        const float* p = reinterpret_cast<const float*>(out->get_cpu_data());
        if (!p) {
            std::cerr << "[ERR] output tensor get_cpu_data() returned nullptr\n";
            return;
        }

        bool has_nan = false, has_inf = false;
        float mn = +1e30f, mx = -1e30f;
        for (int i = 0; i < n; ++i) {
            float v = p[i];
            if (!std::isfinite(v)) {
                has_nan |= std::isnan(v);
                has_inf |= std::isinf(v);
            } else {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        std::cout << "[DBG] output tensor: elems=" << n << " min=" << mn << " max=" << mx << " NaN=" << (has_nan ? "YES" : "no")
                  << " Inf=" << (has_inf ? "YES" : "no") << "\n";

        // 打印前 16 个数
        std::cout << "      first16:";
        for (int i = 0; i < std::min(16, n); ++i)
            std::cout << " " << p[i];
        std::cout << "\n";
    }
}

static const float kArc5Pts[5][2] = {
    { 38.2946f, 51.6963f }, { 73.5318f, 51.5014f }, { 56.0252f, 71.7366f }, { 41.5493f, 92.3655f }, { 70.7299f, 92.2041f },
};

ArcFaceBM::ArcFaceBM(std::shared_ptr<BMNNContext> ctx)
    : m_bmContext(std::move(ctx)) {
    std::cout << "ArcFaceBM ctor .." << std::endl;
}

ArcFaceBM::~ArcFaceBM() {
    // 释放分配的 bm_image
    for (auto& img : m_converto_imgs)
        bm_image_destroy(img);
    for (auto& img : m_planar_imgs)
        bm_image_destroy(img);
    std::cout << "ArcFaceBM dtor ..." << std::endl;
}

int ArcFaceBM::Init(float sim_thresh) {
    m_sim_thresh = sim_thresh;

    // 1) 获取网络
    m_bmNetwork = m_bmContext->network(0);

    // 2) 输入信息
    max_batch = m_bmNetwork->maxBatch();
    auto in_tensor = m_bmNetwork->inputTensor(0);
    auto in_shape = in_tensor->get_shape();  // NCHW
    // 一般为 [1,3,112,112]
    m_net_h = in_shape->dims[2];
    m_net_w = in_shape->dims[3];

    // 3) 输出个数只校验 >=1（ArcFace 就 1 个 512 向量）
    int out_num = m_bmNetwork->outputTensorNum();
    if (out_num < 1) {
        std::cerr << "ArcFace output tensor num invalid: " << out_num << std::endl;
        return -1;
    }

    // 4) 准备 BMCV 图像缓存（对齐到 64 宽度）
    aligned_w_ = FFALIGN(m_net_w, 64);
    printf("ArcFaceBM aligned_w_=%d\n", aligned_w_);
    int strides[3] = { aligned_w_, aligned_w_, aligned_w_ };
    

    m_planar_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);

    // planar: RGB uint8
    for (int i = 0; i < max_batch; ++i) {
        auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_planar_imgs[i], strides);
        if (ret != BM_SUCCESS)
            return -1;
    }

    bm_image_alloc_contiguous_mem(max_batch, m_planar_imgs.data());

    // converto: float32 或 int8（看模型输入 dtype）
    bm_image_data_format_ext out_dtype = DATA_TYPE_EXT_FLOAT32;
    if (in_tensor->get_dtype() == BM_INT8) {
        out_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    /*
    for (int i = 0; i < max_batch; ++i) {
    auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, out_dtype, &m_converto_imgs[i], strides);
    if (ret != BM_SUCCESS)
        return -1;
    }
    */
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, out_dtype, m_converto_imgs.data(), max_batch);
    if (ret != BM_SUCCESS)
        return -1;

    // 5) [-1,1] 归一化： out = alpha*in + beta
    // uint8 [0,255] -> [-1,1] ：alpha=1/127.5, beta=-1
    converto_attr.alpha_0 = 1.0f / 127.5f;
    converto_attr.beta_0 = -1.0f;
    converto_attr.alpha_1 = 1.0f / 127.5f;
    converto_attr.beta_1 = -1.0f;
    converto_attr.alpha_2 = 1.0f / 127.5f;
    converto_attr.beta_2 = -1.0f;

    std::cout << "*** ArcFace init done. max_batch=" << max_batch << ", net=" << m_net_w << "x" << m_net_h << " ***\n";
    return 0;
}

// 把整图 + 5点直接透视到 112×112（不需要先 crop，warpAffine 内部会取正确区域）
cv::Mat ArcFaceBM::AlignBy5PtsRGB(const cv::Mat& bgr, const cvai_pts_t& pts, int out_w, int out_h) {
    // 构造 src/dst
    cv::Mat src(5, 2, CV_32F);
    for (int i = 0; i < 5; ++i) {
        src.at<float>(i, 0) = pts.x[i];
        src.at<float>(i, 1) = pts.y[i];
    }
    cv::Mat dst(5, 2, CV_32F);
    for (int i = 0; i < 5; ++i) {
        dst.at<float>(i, 0) = kArc5Pts[i][0];
        dst.at<float>(i, 1) = kArc5Pts[i][1];
    }

    // 取仿射矩阵（OpenCV 4：第三参是 inliers 的 OutputArray，可省略或传 noArray()）
    cv::Mat M = cv::estimateAffinePartial2D(src, dst);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat aligned(out_h, out_w, CV_8UC3);
    cv::warpAffine(rgb, aligned, M, aligned.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return aligned;
}

bool ArcFaceBM::UploadAndPreprocess(cv::Mat& rgb112, bm_image& out_converto){
    
    /*
    if (rgb112.empty() || rgb112.cols != m_net_w || rgb112.rows != m_net_h || rgb112.channels() != 3)
        return false;
    cv::Mat planes[3];
    cv::split(rgb112, planes);  // R,G, B
    void* host_ptrs[3] = { planes[0].data, planes[1].data, planes[2].data };
    if (bm_image_copy_host_to_device(m_planar_imgs[0], (void**)host_ptrs) != BM_SUCCESS)
        return false;

    std::string dump_dir = "/home/linaro/sophon-demo/sample/SCRFD/cpp/scrfd_bmcv";
    std::string dump_path = GetNextDumpFilename(dump_dir, "dbg_arcface_planar", ".bmimg");
    cv::bmcv::dumpBMImage(&m_planar_imgs[0], dump_path.c_str() );

    cv::Mat dumpMat;
    cv::bmcv::toMAT(&m_planar_imgs[0], dumpMat);
    std::string dump_mat_path = GetNextDumpFilename(dump_dir, "dbg_arcface_planar_mat", ".jpg");
    cv::imwrite(dump_mat_path, dumpMat);

    bm_image test_image;
    cv::bmcv::toBMI(rgb112, &test_image);
    cv::bmcv::toMAT(&test_image, dumpMat);
    std::string dump_mat_path = GetNextDumpFilename(dump_dir, "dbg_arcface_planar_self_mat", ".jpg");
    cv::imwrite(dump_mat_path, dumpMat);

    if (bmcv_image_convert_to(m_bmContext->handle(), 1, converto_attr, m_planar_imgs.data(), m_converto_imgs.data()) != BM_SUCCESS)
        return false;
    auto pr = [](const char* tag, const bm_image& im) {
        int w, h, stride[4];
        w = im.width;
        h = im.height;
        //im.image_format;
        bm_image_get_stride(im, stride);
        std::cout << tag << ": " << w << "x" << h << " fmt=" << im.image_format << " dtype=" << im.data_type << " stride=" << stride[0] << "\n";
    };
    pr("[DBG] planar", m_planar_imgs[0]);
    pr("[DBG] convto", m_converto_imgs[0]);

    out_converto = m_converto_imgs[0];
    */
    std::vector<bm_image> input_images;
    bm_image _bmimg;
    cv::bmcv::toBMI(rgb112, &_bmimg);
    input_images.push_back(_bmimg);
    this->pre_process( input_images );

    return true;
}

//std::vector<float> ArcFaceBM::InferenceOnce(const bm_image& input_converto){
std::vector<float> ArcFaceBM::InferenceOnce(){
    /*
    auto input = m_bmNetwork->inputTensor(0);
    bm_device_mem_t dev_mem;
    bm_image_get_device_mem(input_converto, &dev_mem);
    input->set_device_mem(&dev_mem);
    input->set_shape_by_dim(0, 1);

    // ☆ forward 前检查：bm_image 的数值（抽样）
    DebugCheckBmImageF32RGBPlanar(m_bmContext->handle(), input_converto, "converto@forward");

    // ☆ forward 前检查：input tensor 形状/类型/scale
    DebugCheckTensorInOut(input.get(), nullptr);

    int stride[3];
    bm_image_get_stride(input_converto, stride);
    printf("!!!stride[0]=%d (expected %d)\n", stride[0], 112 * 4);
    */
    int ret = m_bmNetwork->forward();
    CV_Assert(ret == 0);

    std::shared_ptr<BMNNTensor> out = m_bmNetwork->outputTensor(0);
    //std::cout << "[DBG] outputTensor is on device=" << out->is_device() << std::endl;
    // 强制同步 Device → Host
    DebugCheckTensorInOut(nullptr, out.get());
    printf("==--==\n");
    const bm_shape_t* shp = out->get_shape();
    int out_num = 1;
    for (int i = 0; i < shp->num_dims; ++i){
        out_num *= shp->dims[i];
    }
    printf("output tensor num=%d\n", out_num);
    const float* ptr = reinterpret_cast<const float*>(out->get_cpu_data()) ;
    if (!ptr || out_num <= 0)
        return {};
    std::vector<float> feat(out_num);
    std::memcpy(feat.data(), ptr, sizeof(float) * out_num);
    return feat;
}

std::vector<float> ArcFaceBM::Forward112RGB(cv::Mat& rgb112) {
    bm_image conv;
    if (!UploadAndPreprocess(rgb112, conv))
        return {};
    //auto feat = InferenceOnce(conv);
    auto feat = InferenceOnce();
    if (feat.empty())
        return feat;
    double s = 0;
    for (auto v : feat)
        s += double(v) * double(v);
    float n = std::sqrt((float)s);
    if (!(n > 1e-6f) || !std::isfinite(n))
        return {};
    for (auto& v : feat)
        v /= n;
    return feat;
}

std::vector<float> ArcFaceBM::Forward112BGR(const cv::Mat& bgr112){
    cv::Mat rgb;
    cv::cvtColor(bgr112, rgb, cv::COLOR_BGR2RGB);
    return Forward112RGB(rgb);
}

float ArcFaceBM::CosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty())
        return -2.0f;
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    na = std::sqrt(na);
    nb = std::sqrt(nb);
    if (na == 0.f || nb == 0.f)
        return -2.0f;
    return dot / (na * nb);
}

bool ArcFaceBM::BuildGallery(const std::string& root,bm_handle_t& bm_h , Scrfd& detector, float det_conf) {
    gallery_.clear();
    if (!fs::exists(root)) {
        std::cerr << "[BuildGallery] path not exist: " << root << std::endl;
        return false;
    }

    std::vector<std::pair<std::string, std::string>> files;
    for (auto& p : fs::directory_iterator(root)) {
        if (!p.is_directory())
            continue;
        std::string person = p.path().filename().string();
        for (auto& f : fs::directory_iterator(p.path())) {
            if (!f.is_regular_file())
                continue;
            auto ext = f.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                files.emplace_back(person, f.path().string());
            }
        }
    }
    if (files.empty()) {
        std::cerr << "[BuildGallery] no images found under " << root << std::endl;
        return false;
    }

    int ok_cnt = 0;
    for (auto& item : files) {
        const std::string& person = item.first;
        std::string& path = item.second;
        cv::Mat img_bgr = cv::imread(path);
        //if (img_bgr.empty())
        //    continue;
        bm_image bmimg;
        printf("will read bm_image %s \n", path.c_str());
        picDec(bm_h , path.c_str(), bmimg);
        printf("get image %s from picDec\n", path.c_str());
        // === 检测单人脸 ===
        //std::vector<cvai_bbox_t> boxes;
        //bool det_ok = detector.DetectOne(img_bgr, boxes);  // 替换为你项目中的检测调用
        std::vector<bm_image> batch_imgs;
        std::vector<ScrfdBoxVec> batch_boxes;
        batch_imgs.push_back(bmimg);
        int det_ok = detector.Detect(batch_imgs, batch_boxes);
        if (det_ok == 0){
            printf("detector.Detect done\n");
        }else{
            printf("detector.Detect failed\n");
            continue;
        }
        
        //CV_Assert(0 == detector.Detect(batch_imgs, batch_boxes));
        ScrfdBoxVec boxes = batch_boxes[0]; 
        bm_image_destroy(batch_imgs[0]);
        printf("bm_image_destroy done\n");
        if (boxes.empty()) {
            std::cerr << "[Gallery] no face: " << path << std::endl;
            continue;
        }

        // 取最高置信度
        int best_idx = -1;
        float best_score = -1.f;
        for (int i = 0; i < (int)boxes.size(); ++i)
            if (boxes[i].bbox.score > best_score) {
                best_score = boxes[i].bbox.score;
                best_idx = i;
            }

        if (best_score < det_conf) {
            std::cerr << "[Gallery] low conf: " << best_score << " @ " << path << std::endl;
            continue;
        }

        const auto& face = boxes[best_idx];
        if (!face.pts.x || face.pts.size < 5)
            continue;

        // === 5点对齐并前向 ===
        cv::Mat rgb112 = ArcFaceBM::AlignBy5PtsRGB(img_bgr, face.pts, 112, 112);
        printf("AlignBy5PtsRGB done\n");
        if (rgb112.empty())
            continue;
        
        path = path + "_aligned.jpg";

        cv::Mat save_mat;
        cv::cvtColor(rgb112, save_mat, cv::COLOR_RGB2BGR);
        cv::imwrite(path, save_mat);

        std::vector<float> feat = this->Forward112RGB(rgb112);
        std::cout << "feat: ";
        for (size_t i = 0; i < feat.size(); i++) {
            std::cout << feat[i] << " ";
        }
        std::cout << std::endl;

        if (feat.size() != 512)
            continue;

        // === 加入图库 ===
        gallery_.push_back({ person, feat });
        ok_cnt++;
        std::cout << "[Gallery] Added: " << person << " (" << fs::path(path).filename().string() << "), conf=" << best_score << std::endl;
    }

    std::cout << "[BuildGallery] completed: " << ok_cnt << " faces added. total gallery size=" << gallery_.size() << std::endl;

    return !gallery_.empty();
}

bool ArcFaceBM::MatchGallery(const std::vector<float>& feat, std::string& best_name, float& best_sim) {
    best_sim = -1.f;
    best_name = "Unknown";

    if (gallery_.empty()){
        printf("gallery_ is empty\n");
        return false;
    }
    if (feat.size() != 512){
        printf("feat size is not 512\n");
        return false;
    }

    for (const auto& g : gallery_) {
        float s = CosineSim(feat, g.feat);
        if (s > best_sim) {
            best_sim = s;
            best_name = g.name;
        }
    }
    return true;
}

int ArcFaceBM::pre_process(const std::vector<bm_image>& images) {
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();
  // 1. resize image
  int ret = 0;
  for (int i = 0; i < image_n; ++i) {
    bm_image image1 = images[i];
    bm_image image_aligned;
    bool need_copy = image1.width & (64 - 1);
    if (need_copy) {
      int stride1[3], stride2[3];
      bm_image_get_stride(image1, stride1);
      stride2[0] = FFALIGN(stride1[0], 64);
      stride2[1] = FFALIGN(stride1[1], 64);
      stride2[2] = FFALIGN(stride1[2], 64);
      bm_image_create(m_bmContext->handle(), image1.height, image1.width,
                      image1.image_format, image1.data_type, &image_aligned,
                      stride2);

      bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
      bmcv_copy_to_atrr_t copyToAttr;
      memset(&copyToAttr, 0, sizeof(copyToAttr));
      copyToAttr.start_x = 0;
      copyToAttr.start_y = 0;
      copyToAttr.if_padding = 1;
      bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, image1,
                         image_aligned);
    } else {
      image_aligned = image1;
    }
#if USE_ASPECT_RATIO
    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(images[i].width, images[i].height,
                                          m_net_w, m_net_h, &isAlignWidth);
    bmcv_padding_atrr_t padding_attr;
    memset(&padding_attr, 0, sizeof(padding_attr));
    padding_attr.dst_crop_sty = 0;
    padding_attr.dst_crop_stx = 0;
    padding_attr.padding_b = 114;
    padding_attr.padding_g = 114;
    padding_attr.padding_r = 114;
    padding_attr.if_memset = 1;
    if (isAlignWidth) {
      padding_attr.dst_crop_h = images[i].height * ratio;
      padding_attr.dst_crop_w = m_net_w;

      int ty1 = (int)((m_net_h - padding_attr.dst_crop_h) / 2);
      padding_attr.dst_crop_sty = ty1;
      padding_attr.dst_crop_stx = 0;
    } else {
      padding_attr.dst_crop_h = m_net_h;
      padding_attr.dst_crop_w = images[i].width * ratio;

      int tx1 = (int)((m_net_w - padding_attr.dst_crop_w) / 2);
      padding_attr.dst_crop_sty = 0;
      padding_attr.dst_crop_stx = tx1;
    }

    bmcv_rect_t crop_rect{0, 0, image1.width, image1.height};
    auto ret = bmcv_image_vpp_convert_padding(
        m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
        &padding_attr, &crop_rect, BMCV_INTER_NEAREST);
#else
    auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i],
                                      &m_planar_imgs[i]);
#endif
    assert(BM_SUCCESS == ret);

#if DUMP_FILE
    cv::Mat resized_img;
    cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
    std::string fname = cv::format("resized_img_%d.jpg", i);
    cv::imwrite(fname, resized_img);
#endif
    if (need_copy) bm_image_destroy(image_aligned);
  }

  // 2. converto
  ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr,
                              m_planar_imgs.data(), m_converto_imgs.data());
  CV_Assert(ret == 0);

  // 3. attach to tensor
  if (image_n != max_batch) image_n = m_bmNetwork->get_nearest_batch(image_n);
  bm_device_mem_t input_dev_mem;
  bm_image_get_contiguous_device_mem(image_n, m_converto_imgs.data(),
                                     &input_dev_mem);
  input_tensor->set_device_mem(&input_dev_mem);
  input_tensor->set_shape_by_dim(0, image_n);  // set real batch number
  return 0;
}

float ArcFaceBM::get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h,
                                     bool* pIsAligWidth) {
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w) {
    *pIsAligWidth = true;
    ratio = r_w;
  } else {
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}