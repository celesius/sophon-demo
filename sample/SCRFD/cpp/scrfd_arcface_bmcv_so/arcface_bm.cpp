#include "arcface_bm.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <numeric>  // for inner_product
#include <opencv2/opencv.hpp>
#include "ff_decode.hpp"

#include <regex>
#include <string>

namespace fs = std::filesystem;

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


// 辅助：用 5点最小二乘解 2x3 仿射矩阵
static void SolveAffine2x3_5pt(const float src[5][2], const float dst[5][2], float M[2][3]) {
    // 正规方程：A^T A theta = A^T b
    // theta = [a b tx c d ty]^T
    double ATA[6][6] = {0}, ATb[6] = {0};
    auto accum = [&](double x, double y, double xp, double yp) {
        double rowx[6] = {x, y, 1.0, 0, 0, 0};
        double rowy[6] = {0, 0, 0,   x, y, 1.0};
        // 累加
        for(int r=0;r<6;++r){
            for(int c=0;c<6;++c){
                ATA[r][c] += rowx[r]*rowx[c] + rowy[r]*rowy[c];
            }
        }
        for(int r=0;r<6;++r){
            ATb[r] += rowx[r]*xp + rowy[r]*yp;
        }
    };
    for(int i=0;i<5;++i) accum(src[i][0], src[i][1], dst[i][0], dst[i][1]);

    // 解 6x6 线性方程（高斯消元）
    int n=6;
    for(int i=0;i<n;++i){
        // 选主元
        int piv=i;
        for(int r=i+1;r<n;++r) if(fabs(ATA[r][i])>fabs(ATA[piv][i])) piv=r;
        if(piv!=i){ for(int c=i;c<n;++c) std::swap(ATA[i][c],ATA[piv][c]); std::swap(ATb[i],ATb[piv]); }
        double div=ATA[i][i]+1e-12;
        for(int c=i;c<n;++c) ATA[i][c]/=div; ATb[i]/=div;
        for(int r=0;r<n;++r) if(r!=i){
            double factor=ATA[r][i];
            for(int c=i;c<n;++c) ATA[r][c]-=factor*ATA[i][c];
            ATb[r]-=factor*ATb[i];
        }
    }
    double a=ATb[0], b=ATb[1], tx=ATb[2], c=ATb[3], d=ATb[4], ty=ATb[5];
    M[0][0]=float(a); M[0][1]=float(b); M[0][2]=float(tx);
    M[1][0]=float(c); M[1][1]=float(d); M[1][2]=float(ty);
}


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


inline bool invert_affine_2x3(const float M[2][3], float Minv[2][3]) {
    double a=M[0][0], b=M[0][1], tx=M[0][2];
    double c=M[1][0], d=M[1][1], ty=M[1][2];
    double det = a*d - b*c;
    if (fabs(det) < 1e-12) return false;

    double ia =  d/det, ib = -b/det;
    double ic = -c/det, id =  a/det;
    double itx = -(ia*tx + ib*ty);
    double ity = -(ic*tx + id*ty);

    Minv[0][0]=float(ia);  Minv[0][1]=float(ib);  Minv[0][2]=float(itx);
    Minv[1][0]=float(ic);  Minv[1][1]=float(id);  Minv[1][2]=float(ity);
    return true;
}

int ArcFaceBM::AlignBy5PtsBMI(bm_image& src_bmi, const cvai_pts_t& pts, bm_image& dst_rgb112) {
    //int strides[3] = { FFALIGN(112, 64), FFALIGN(112, 64), FFALIGN(112, 64) };
    //int ssrc[3] = { FFALIGN(src_bmi.width,64), FFALIGN(src_bmi.width,64), FFALIGN(src_bmi.width,64) };

    // 2) 组 5点
    float src5[5][2], dst5[5][2];
    for (int i=0;i<5;++i){ src5[i][0]=pts.x[i]; src5[i][1]=pts.y[i]; }
    for (int i=0;i<5;++i){ dst5[i][0]=kArc5Pts[i][0]; dst5[i][1]=kArc5Pts[i][1]; }

    // 3) 求 2x3 仿射
    float M[2][3];
    SolveAffine2x3_5pt(src5, dst5, M);

    // 1) 中间src图像 
    bm_image src_aligned;
    bm_image_create(m_bmContext->handle(), src_bmi.height, src_bmi.width, FORMAT_RGB_PLANAR, src_bmi.data_type, &src_aligned, NULL);
    bm_image_alloc_dev_mem(src_aligned, BMCV_IMAGE_FOR_IN);
    bmcv_image_storage_convert(m_bmContext->handle(), 1, &src_bmi, &src_aligned);

    // 4) 最终输出图像 
    //auto ret = bm_image_create(m_bmContext->handle(), src_aligned.height, src_aligned.width, src_aligned.image_format, src_aligned.data_type, &dst_rgb112, NULL);
    auto ret = bm_image_create(m_bmContext->handle(), 112, 112, src_aligned.image_format, src_aligned.data_type, &dst_rgb112, NULL);
    if (ret != BM_SUCCESS) return -1;
    //bm_image_alloc_dev_mem(dst_rgb112, BMCV_HEAP1_ID);
    if(bm_image_alloc_dev_mem(dst_rgb112, BMCV_IMAGE_FOR_IN) != BM_SUCCESS){
        printf("[ArcFace] alloc dst_rgb112 dev mem failed");
        bm_image_destroy(dst_rgb112);
        return -1;
    }

    /*
    if (!(src_bmi.image_format == FORMAT_RGB_PLANAR && src_bmi.data_type == DATA_TYPE_EXT_1N_BYTE)) {
        bm_image tmp;
        int s2[3] = { FFALIGN(src_bmi.width,64), FFALIGN(src_bmi.width,64), FFALIGN(src_bmi.width,64) };
        bm_image_create(m_bmContext->handle(), src_bmi.height, src_bmi.width, FORMAT_RGB_PLANAR,
                        DATA_TYPE_EXT_1N_BYTE, &tmp, s2);
        bm_image_alloc_dev_mem(tmp, BMCV_HEAP1_ID);
        // 颜色/打包转换
        bmcv_image_storage_convert(m_bmContext->handle(), 1, &src_bmi, &tmp);
        src_aligned = tmp;
    }
    */

    float IM[2][3];
    if (!invert_affine_2x3(M, IM)) {
        std::cerr << "AlignBy5PtsBMI: invert_affine_2x3 failed\n";
        //if (src_aligned.data != src_bmi.data) bm_image_destroy(src_aligned);
        return -1;
    }
    // 5) 仿射到 112×112
    bmcv_affine_image_matrix aff_img{};
    aff_img.matrix_num = 1;

    bmcv_affine_matrix mat{};
    // 将计算得到的 2x3 仿射矩阵 M 赋值给 bmcv_affine_matrix 的成员
    mat.m[0] = IM[0][0];                                   // 第一行第一列
    mat.m[1] = IM[0][1];                                   // 第一行第二列
    mat.m[2] = IM[0][2];                                   // 第一行第三列（平移量 tx）
    mat.m[3] = IM[1][0];                                   // 第二行第一列
    mat.m[4] = IM[1][1];                                   // 第二行第二列
    mat.m[5] = IM[1][2];                                   // 第二行第三列（平移量 ty）
    aff_img.matrix = &mat;

    //bm_status_t result_write =  bm_image_write_to_bmp(src_aligned, "/home/linaro/src_aligned.bmp");
    //printf("II [Face3D][DEBUG] bm_image_write_to_bmp src_aligned ret=%d\n", result_write );

    ret = bmcv_image_warp_affine(m_bmContext->handle(), 1, &aff_img, &src_aligned, &dst_rgb112, 0);
    if (ret != BM_SUCCESS) {
        printf("[ArcFace] warp_affine failed, ret=%d\n", ret);
        return -1;
    }

    //result_write =  bm_image_write_to_bmp(dst_rgb112 , "/home/linaro/dst_rgb112.bmp");
    //printf("II [Face3D][DEBUG] bm_image_write_to_bmp dst_rgb112.bmp ret=%d\n", result_write );
    
    return (ret==BM_SUCCESS)?0:-1;
}


// 把整图 + 5点直接透视到 112×112（不需要先 crop，warpAffine 内部会取正确区域）
/*
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
*/

//bool ArcFaceBM::UploadAndPreprocess(cv::Mat& rgb112, bm_image& out_converto){
bool ArcFaceBM::UploadAndPreprocess(bm_image& rgb112, bm_image& out_converto){
    
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
    //bm_image _bmimg;
    //cv::bmcv::toBMI(rgb112, &_bmimg);
    input_images.push_back(rgb112);
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
    //DebugCheckTensorInOut(nullptr, out.get());
    //printf("==--==\n");
    const bm_shape_t* shp = out->get_shape();
    int out_num = 1;
    for (int i = 0; i < shp->num_dims; ++i){
        out_num *= shp->dims[i];
    }
    //printf("output tensor num=%d\n", out_num);
    const float* ptr = reinterpret_cast<const float*>(out->get_cpu_data()) ;
    if (!ptr || out_num <= 0)
        return {};
    std::vector<float> feat(out_num);
    std::memcpy(feat.data(), ptr, sizeof(float) * out_num);
    return feat;
}

std::vector<float> ArcFaceBM::Forward112RGB(bm_image& rgb112) {
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

std::vector<float> ArcFaceBM::Forward112BGR(bm_image & bgr112){
    //cv::Mat rgb;
    //cv::cvtColor(bgr112, rgb, cv::COLOR_BGR2RGB);
    return Forward112RGB(bgr112);
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
        
        //if (img_bgr.empty())
        //    continue;
        bm_image bmimg;
        printf("will read bm_image %s \n", path.c_str());
        //picDec(bm_h , path.c_str(), bmimg);
        bm_handle_t hh = m_bmContext->handle();
        picDec(hh, path.c_str(), bmimg);
        printf("get image %s from picDec\n", path.c_str());
        //bm_status_t rrrr =  bm_image_write_to_bmp(bmimg, "/home/linaro/bmimg.bmp");
        //printf("II [Face3D][DEBUG] bm_image_write_to_bmp bmimg ret=%d\n", rrrr);
        // === 检测单人脸 ===
        //std::vector<cvai_bbox_t> boxes;
        //bool det_ok = detector.DetectOne(img_bgr, boxes);  
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
        printf("bm_image_destroy done\n");
        if (boxes.empty()) {
            std::cerr << "[Gallery] no face: " << path << std::endl;
            continue;
        }else{
            printf("detected %lu faces in %s\n", boxes.size(), path.c_str());
        }

        // 取最高置信度
        int best_idx = -1;
        float best_score = -1.f;
        for (int i = 0; i < (int)boxes.size(); ++i){
            if (boxes[i].bbox.score > best_score) {
                best_score = boxes[i].bbox.score;
                best_idx = i;
            }
        }
        printf("best_score=%f , det_conf=%f\n", best_score, det_conf);
        if (best_score < det_conf) {
            std::cerr << "[Gallery] low conf: " << best_score << " @ " << path << std::endl;
            continue;
        }
        const auto& face = boxes[best_idx];
        if (!face.pts.x || face.pts.size < 5)
            continue;
        printf("face pts got hhhhhhhh \n");
        // === 5点对齐并前向 ===
        bm_image rgb112;
        //bm_status_t rrrr =  bm_image_write_to_bmp(bmimg, "/home/linaro/bmimg.bmp");
        //printf("II [Face3D][DEBUG] bm_image_write_to_bmp bmimg ret=%d\n", rrrr);
        int rrr =  AlignBy5PtsBMI(bmimg, face.pts, rgb112);
        //rrrr =  bm_image_write_to_bmp(rgb112, "/home/linaro/rgb112.bmp");
        //printf("II [Face3D][DEBUG] bm_image_write_to_bmp rgb112 ret=%d\n", rrrr);

        printf("AlignBy5PtsRGB done\n");
        bm_image_destroy(batch_imgs[0]);
        //if (rgb112.empty())
        //    continue;
        /*
        path = path + "_aligned.jpg";
        cv::Mat save_mat;
        cv::cvtColor(rgb112, save_mat, cv::COLOR_RGB2BGR);
        cv::imwrite(path, save_mat);
        */ 
        std::vector<float> feat = this->Forward112BGR(rgb112);

        bm_image_destroy(rgb112);
        /*
        std::cout << "feat: ";
        for (size_t i = 0; i < feat.size(); i++) {
            std::cout << feat[i] << " ";
        }
        std::cout << std::endl;
        */
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


/**
 * @brief ArcFaceBM 批量推理接口
 * @param input 输入图像列表，要求每个图像已经是全此缓存 RGB
 * @param output 输出arcface_data 列表, 每个列表中包含人名和sim结果
 * @return 成功返回 true，失败返回 false
 * 函数内部实现整个人脸识别流程：预处理、前向、特征归一化、与图库比对，最终输出识别结果
*/
bool ArcFaceBM::arcface_inference(bm_image& input, const ScrfdBoxVec& detected_boxes, std::vector<arcface_data>& output)
{
    output.clear();
    //获得矫正后的人脸图像列表
    for (const auto& box : detected_boxes) {
        bm_image rgb112;
        int rrr =  AlignBy5PtsBMI(input, box.pts, rgb112);
        if (rrr != 0) {
            std::cerr << "[arcface_inference] AlignBy5PtsBMI failed\n";
            continue;
        }
        arcface_data afd;
        afd.bbox = box.bbox;
        //output.push_back(afd);

        std::vector <float> feat = this->Forward112RGB(rgb112);
        bm_image_destroy(rgb112);
        if (feat.size() != 512) {
            std::cerr << "[arcface_inference] Forward112RGB failed\n";
            continue;
        }
        // 与图库比对
        std::string best_name;
        float best_sim;
        if (!this->MatchGallery(feat, best_name, best_sim)) {
            std::cerr << "[arcface_inference] MatchGallery failed\n";
            continue;
        }
        afd.person_name = best_name;
        afd.sim = best_sim;
        output.push_back(afd);
    }

    return true; 
}