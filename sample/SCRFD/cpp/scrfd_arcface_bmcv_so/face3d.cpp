#include "face3d.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <cassert>
#include <string>


#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include "bmcv_api.h"


using namespace face3d;

// ========== 构造/创建 ==========
Face3D::Face3D(std::shared_ptr<BMNNContext> ctx):m_bmContext(ctx) {
    handle_ = m_bmContext->handle();
    std::cout << "Face3D ctor .." << std::endl;
}
Face3D::~Face3D() {
    std::cout << "Face3D dtor ..." << std::endl;
    /*
    bm_image_free_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_free_contiguous_mem(max_batch, m_converto_imgs.data());
    for (int i = 0; i < max_batch; i++) {
        bm_image_destroy(m_converto_imgs[i]);
        bm_image_destroy(m_resized_imgs[i]);
    }
*/
}
static bool load_meanshape_txt(const std::string& p, std::array<Point3f, 68>& m) {
    std::ifstream ifs(p);
    if (!ifs.is_open())
        return false;
    for (int i = 0; i < 68; i++) {
        float x, y, z;
        if (!(ifs >> x >> y >> z))
            return false;
        m[i] = { x, y, z };
    }
    return true;
}

static void dump_bm_image_info(const char* tag, const bm_image& img){
    int w=0,h=0;
    w = img.width;
    h = img.height;
    int planes=0;
    planes = bm_image_get_plane_num(img);
    int stride[4]={0};
    bm_image_get_stride(img, stride);

    printf("%s: %dx%d fmt=%d dtype=%d planes=%d\n",
           tag, w, h, (int)img.image_format, (int)img.data_type, planes);
    bm_device_mem_t mem[4]; memset(mem,0,sizeof(mem));
    bm_image_get_device_mem(img, mem);
    for(int i=0;i<planes;++i){
        printf("  plane[%d] stride=%d dev=0x%llx\n",
               i, stride[i], (unsigned long long)mem[i].u.device.device_addr);
    }
}

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
    if (bm_image_create(target_handle, w, h, fmt.image_format, dtype, &dst_same_handle) != BM_SUCCESS)
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

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>

static std::string now_tag() {
    using namespace std::chrono;
    auto t = system_clock::now().time_since_epoch();
    return std::to_string(duration_cast<milliseconds>(t).count());
}

static void dump_vec_bin(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

static void dump_vec_txt_head(const std::string& path, const std::vector<float>& v, int n = 64) {
    std::ofstream f(path);
    f << std::setprecision(6) << std::fixed;
    int m = std::min<int>(n, v.size());
    for (int i = 0; i < m; ++i)
        f << v[i] << (i + 1 == m ? '\n' : ' ');
}

static void scan_stats(const std::vector<float>& v) {
    size_t nonfinite = 0;
    double s = 0.0, s2 = 0.0;
    float mn = +1e30f, mx = -1e30f;
    for (float x : v) {
        if (!std::isfinite(x)) {
            nonfinite++;
            continue;
        }
        mn = std::min(mn, x);
        mx = std::max(mx, x);
        s += x;
        s2 += (double)x * x;
    }
    double n = (double)v.size();
    double mean = s / n;
    double var = std::max(0.0, s2 / n - mean * mean);
    fprintf(stderr, "[fc1] size=%zu nonfinite=%zu min=%.6f max=%.6f mean=%.6f std=%.6f\n", v.size(), nonfinite, mn, mx, mean, std::sqrt(var));
}

void Face3D::debug_dump_fc1_and_crop(const std::vector<float>& fc1, const bm_image& crop_u8_bgr, const Affine2x3& M, const Affine2x3& IM) {
    std::string tag = now_tag();

    // 1) 数值体检 & 前几个值
    scan_stats(fc1);
    dump_vec_txt_head("/tmp/fc1_head_" + tag + ".txt", fc1, 64);
    dump_vec_bin("/tmp/fc1_" + tag + ".bin", fc1);

    // 2) 存裁剪图（PPM，避免引入 OpenCV）
    //    crop_u8_bgr：应为 192x192, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE
    int w = crop_u8_bgr.width, h = crop_u8_bgr.height;
    std::vector<uint8_t> host;
    host.resize(w * h * 3);
    void* planes[3] = { host.data(), nullptr, nullptr };
    // D2H
    if (bm_image_copy_device_to_host(crop_u8_bgr, planes) == BM_SUCCESS) {
        std::ofstream f("/tmp/crop_" + tag + ".ppm", std::ios::binary);
        // PPM header
        f << "P6\n" << w << " " << h << "\n255\n";
        // BGR -> RGB 逐像素
        for (int i = 0; i < w * h; ++i) {
            uint8_t b = host[3 * i + 0], g = host[3 * i + 1], r = host[3 * i + 2];
            f.put((char)r);
            f.put((char)g);
            f.put((char)b);
        }
    }

    // 3) 保存仿射矩阵
    {
        std::ofstream f("/tmp/affine_" + tag + ".txt");
        f << std::setprecision(6) << std::fixed;
        f << "M: " << M.a00 << " " << M.a01 << " " << M.a02 << " | " << M.a10 << " " << M.a11 << " " << M.a12 << "\n";
        f << "IM: " << IM.a00 << " " << IM.a01 << " " << IM.a02 << " | " << IM.a10 << " " << IM.a11 << " " << IM.a12 << "\n";
    }
}

int Face3D::Init(const Config& cfg) {
    cfg_ = cfg;

    bool load_ret = load_meanshape(cfg_.meanshape_txt);
    if (load_ret == false)
    {
        std::cerr << "Face3D::Init load meanshape failed from " << cfg_.meanshape_txt << std::endl;
        return -1;
    }

    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    // 2. get input
    max_batch = m_bmNetwork->maxBatch();
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    // 3. get output
    output_num = m_bmNetwork->outputTensorNum();
    assert(output_num == 1);
    min_dim = m_bmNetwork->outputTensor(0)->get_shape()->num_dims;

    // 4. initialize bmimages
    m_resized_imgs.resize(max_batch);
    m_converto_imgs.resize(max_batch);
    // some API only accept bm_image whose stride is aligned to 64
    int aligned_net_w = FFALIGN(m_net_w, 64);
    int strides[3] = { aligned_net_w, aligned_net_w, aligned_net_w };
    for (int i = 0; i < max_batch; i++) {
        auto ret = bm_image_create(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &m_resized_imgs[i], strides);
        assert(BM_SUCCESS == ret);
    }
    bm_image_alloc_contiguous_mem(max_batch, m_resized_imgs.data());
    bm_image_data_format_ext img_dtype = DATA_TYPE_EXT_FLOAT32;
    if (tensor->get_dtype() == BM_INT8) {
        img_dtype = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    auto ret = bm_image_create_batch(m_bmContext->handle(), m_net_h, m_net_w, FORMAT_RGB_PLANAR, img_dtype, m_converto_imgs.data(), max_batch);
    assert(BM_SUCCESS == ret);

    // 5.converto
    float input_scale = tensor->get_scale();
    //input_scale = input_scale * 1.0 / 255.f;
    input_scale = 1.0;
    converto_attr.alpha_0 = input_scale;
    converto_attr.beta_0 = 0.0;
    converto_attr.alpha_1 = input_scale;
    converto_attr.beta_1 = 0.0;
    converto_attr.alpha_2 = input_scale;
    converto_attr.beta_2 = 0.0;

    return 0;
}

// ========== 数学工具 ==========
void Face3D::invertAffine(const Affine2x3& M, Affine2x3& IM) {
    double det = (double)M.a00 * M.a11 - (double)M.a01 * M.a10;
    if (std::abs(det) < 1e-12)
        det = (M.a00 >= 0 ? 1 : -1) * 1e-12;
    double inv = 1.0 / det;
    IM.a00 = (float)(M.a11 * inv);
    IM.a01 = (float)(-M.a01 * inv);
    IM.a10 = (float)(-M.a10 * inv);
    IM.a11 = (float)(M.a00 * inv);
    IM.a02 = (float)(-(IM.a00 * M.a02 + IM.a01 * M.a12));
    IM.a12 = (float)(-(IM.a10 * M.a02 + IM.a11 * M.a12));
}
void Face3D::affineTransformPoints(const Affine2x3& M, const Point2f* in, Point2f* out, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i].x, y = in[i].y;
        out[i].x = M.a00 * x + M.a01 * y + M.a02;
        out[i].y = M.a10 * x + M.a11 * y + M.a12;
    }
}
void Face3D::eulerFromR(const Mat3d& R, float& pitch_deg, float& yaw_deg, float& roll_deg) {
    double rx = std::atan2(R.m[2][1], R.m[2][2]);
    double ry = std::atan2(-R.m[2][0], std::sqrt(R.m[2][1] * R.m[2][1] + R.m[2][2] * R.m[2][2]));
    double rz = std::atan2(R.m[1][0], R.m[0][0]);
    pitch_deg = (float)(rx * 180.0 / M_PI);
    yaw_deg = (float)(ry * 180.0 / M_PI);
    roll_deg = (float)(rz * 180.0 / M_PI);
}
Mat3d Face3D::mul(const Mat3d& A, const Mat3d& B) {
    Mat3d C {};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++)
                s += A.m[i][k] * B.m[k][j];
            C.m[i][j] = s;
        }
    return C;
}

/*
static const char* fmt_name(bm_image_format_enum f) {
    switch (f) {
        case FORMAT_BGR_PACKED:
            return "BGR_PACKED";
        case FORMAT_RGB_PACKED:
            return "RGB_PACKED";
        case FORMAT_NV12:
            return "NV12";
        case FORMAT_NV21:
            return "NV21";
        case FORMAT_YUV420P:
            return "YUV420P";
        case FORMAT_RGB_PLANAR:
            return "RGB_PLANAR";
        case FORMAT_BGR_PLANAR:
            return "BGR_PLANAR";
        default:
            return "UNKNOWN";
    }
}*/
static const char* dtype_name(bm_image_data_format_ext d) {
    switch (d) {
        case DATA_TYPE_EXT_1N_BYTE:
            return "U8";
        case DATA_TYPE_EXT_FLOAT32:
            return "F32";
        case DATA_TYPE_EXT_1N_BYTE_SIGNED:
            return "S8";
        default:
            return "OTHER";
    }
}

Vec3d Face3D::add(const Vec3d& a, const Vec3d& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Vec3d Face3D::sub(const Vec3d& a, const Vec3d& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Vec3d Face3D::smul(double s, const Vec3d& a) { return { s * a.x, s * a.y, s * a.z }; }
double Face3D::dot(const Vec3d& a, const Vec3d& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
void Face3D::jacobiEigenSymmetric3x3(const Mat3d& A, Mat3d& V, double d[3]) {
    Mat3d D = A;
    V = Mat3d::eye();
    const int IT = 50;
    for (int it = 0; it < IT; ++it) {
        int p = 0, q = 1;
        double m = fabs(D.m[0][1]);
        if (fabs(D.m[0][2]) > m) {
            p = 0;
            q = 2;
            m = fabs(D.m[0][2]);
        }
        if (fabs(D.m[1][2]) > m) {
            p = 1;
            q = 2;
            m = fabs(D.m[1][2]);
        }
        if (m < 1e-12)
            break;
        double app = D.m[p][p], aqq = D.m[q][q], apq = D.m[p][q];
        double phi = 0.5 * atan2(2 * apq, (aqq - app));
        double c = cos(phi), s = sin(phi);
        for (int k = 0; k < 3; k++) {
            double dpk = D.m[p][k], dqk = D.m[q][k];
            D.m[p][k] = c * dpk - s * dqk;
            D.m[q][k] = s * dpk + c * dqk;
        }
        for (int k = 0; k < 3; k++) {
            double dkp = D.m[k][p], dkq = D.m[k][q];
            D.m[k][p] = c * dkp - s * dkq;
            D.m[k][q] = s * dkp + c * dkq;
        }
        for (int k = 0; k < 3; k++) {
            double vkp = V.m[k][p], vkq = V.m[k][q];
            V.m[k][p] = c * vkp - s * vkq;
            V.m[k][q] = s * vkp + c * vkq;
        }
    }
    d[0] = D.m[0][0];
    d[1] = D.m[1][1];
    d[2] = D.m[2][2];
}
void Face3D::svd3x3_AT_A(const Mat3d& A, Mat3d& U, double s[3], Mat3d& Vt) {
    Mat3d AT {};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            AT.m[r][c] = A.m[c][r];
    Mat3d B = mul(AT, A);
    Mat3d V;
    double ev[3];
    jacobiEigenSymmetric3x3(B, V, ev);
    int idx[3] = { 0, 1, 2 };
    auto ord = [&](int i, int j) { return ev[idx[i]] > ev[idx[j]]; };
    if (!ord(0, 1))
        std::swap(idx[0], idx[1]);
    if (!ord(1, 2))
        std::swap(idx[1], idx[2]);
    if (!ord(0, 1))
        std::swap(idx[0], idx[1]);
    Mat3d Vsorted {};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            Vsorted.m[r][c] = V.m[r][idx[c]];
    for (int i = 0; i < 3; i++)
        s[i] = std::sqrt(std::max(0.0, ev[idx[i]]));
    Mat3d Sinv = Mat3d::eye();
    for (int i = 0; i < 3; i++)
        Sinv.m[i][i] = (s[i] > 1e-12) ? 1.0 / s[i] : 0.0;
    U = mul(mul(A, Vsorted), Sinv);
    Vt = Mat3d {};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            Vt.m[r][c] = Vsorted.m[c][r];
}
void Face3D::umeyama_3d3d(const Vec3d* X, const Vec3d* Y, int n, float& s, Mat3d& R, Vec3d& t) {
    Vec3d muX { 0, 0, 0 }, muY { 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        muX = add(muX, smul(1.0 / n, X[i]));
        muY = add(muY, smul(1.0 / n, Y[i]));
    }
    Mat3d Sigma {};
    double varX = 0.0;
    for (int i = 0; i < n; i++) {
        Vec3d x = { X[i].x - muX.x, X[i].y - muX.y, X[i].z - muX.z };
        Vec3d y = { Y[i].x - muY.x, Y[i].y - muY.y, Y[i].z - muY.z };
        Sigma.m[0][0] += y.x * x.x;
        Sigma.m[0][1] += y.x * x.y;
        Sigma.m[0][2] += y.x * x.z;
        Sigma.m[1][0] += y.y * x.x;
        Sigma.m[1][1] += y.y * x.y;
        Sigma.m[1][2] += y.y * x.z;
        Sigma.m[2][0] += y.z * x.x;
        Sigma.m[2][1] += y.z * x.y;
        Sigma.m[2][2] += y.z * x.z;
        varX += x.x * x.x + x.y * x.y + x.z * x.z;
    }
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            Sigma.m[r][c] /= n;
    varX /= n;

    Mat3d U, Vt;
    double sv[3];
    svd3x3_AT_A(Sigma, U, sv, Vt);
    Mat3d UVt = mul(U, Vt);
    double detUVt = UVt.m[0][0] * (UVt.m[1][1] * UVt.m[2][2] - UVt.m[1][2] * UVt.m[2][1])
        - UVt.m[0][1] * (UVt.m[1][0] * UVt.m[2][2] - UVt.m[1][2] * UVt.m[2][0]) + UVt.m[0][2] * (UVt.m[1][0] * UVt.m[2][1] - UVt.m[1][1] * UVt.m[2][0]);
    Mat3d D = Mat3d::eye();
    if (detUVt < 0)
        D.m[2][2] = -1.0;
    R = mul(mul(U, D), Vt);
    double trDS = D.m[0][0] * sv[0] + D.m[1][1] * sv[1] + D.m[2][2] * sv[2];
    s = (float)(trDS / (varX > 1e-12 ? varX : 1e-12));
    Vec3d RmuX { R.m[0][0] * muX.x + R.m[0][1] * muX.y + R.m[0][2] * muX.z, R.m[1][0] * muX.x + R.m[1][1] * muX.y + R.m[1][2] * muX.z,
                 R.m[2][0] * muX.x + R.m[2][1] * muX.y + R.m[2][2] * muX.z };
    t = { muY.x - s * RmuX.x, muY.y - s * RmuX.y, muY.z - s * RmuX.z };
}

bool Face3D::align_to_192_bm(bm_image& src_bgr_in, const FaceBox& box,
                             bm_image& aligned_rgb_f32,  // 输出：RGB planar F32 (192x192)
                             Affine2x3& M, Affine2x3& IM) const {
    
    //dump_bm_image_info("[Face3D] src_bgr_in", src_bgr_in);
    
    const int S = cfg_.input_size;  // 192
    // 1) 计算仿射 M、IM（与 InsightFace 一致）
    const float cx = 0.5f * (box.x1 + box.x2);
    const float cy = 0.5f * (box.y1 + box.y2);
    const float w = (box.x2 - box.x1);
    const float h = (box.y2 - box.y1);
    const float scale = S / (std::max(w, h) * 1.5f);

    M.a00 = scale;
    M.a01 = 0.0f;
    M.a02 = S * 0.5f - cx * scale;
    M.a10 = 0.0f;
    M.a11 = scale;
    M.a12 = S * 0.5f - cy * scale;
    invertAffine(M, IM);

    // 2) 确保与本 sdk 的 handle 一致
    /*
    bm_image src_same;
    if (!ensure_image_on_handle(const_cast<bm_image&>(src_bgr_in), handle_, src_same)) {
        printf("[Face3D] ensure_image_on_handle failed\n");
        return false;
    }
    */

    // 3) 把输入统一成 RGB_PLANAR + U8（warp_affine 支持最好的格式）
    bm_image_format_info fmt_rgbp;
    fmt_rgbp.image_format = FORMAT_RGB_PLANAR;
    bm_image_data_format_ext dtype_u8 = DATA_TYPE_EXT_1N_BYTE;

    bm_image src_u8_rgb;  // 中间图（与 src_same 同尺寸）
    if (bm_image_create(handle_, src_bgr_in.height, src_bgr_in.width, fmt_rgbp.image_format, dtype_u8, &src_u8_rgb) != BM_SUCCESS) {
        printf("[Face3D] bm_image_create src_u8_rgb failed\n");
        return false;
    }
    if (bm_image_alloc_dev_mem(src_u8_rgb, BMCV_HEAP1_ID) != BM_SUCCESS) {
        printf("[Face3D] alloc src_u8_rgb dev mem failed");
        bm_image_destroy(src_u8_rgb);
        return false;
    }

    // 将 src_same → RGB_PLANAR U8（仅格式/通道重排，不做缩放）
    bmcv_convert_to_attr to_u8 {};
    to_u8.alpha_0 = 1.f;
    to_u8.beta_0 = 0.f;
    to_u8.alpha_1 = 1.f;
    to_u8.beta_1 = 0.f;
    to_u8.alpha_2 = 1.f;
    to_u8.beta_2 = 0.f;

    bm_status_t ret = bmcv_image_convert_to(handle_, 1, to_u8, const_cast<bm_image*>(&src_bgr_in ), &src_u8_rgb);
    if (ret != BM_SUCCESS) {
        printf("[Face3D] convert_to -> RGB_PLANAR U8 failed, ret=%d\n", ret);
        bm_image_destroy(src_u8_rgb);
        return false;
    }

    // 4) 准备 warp 目标：RGB_PLANAR + U8, 192x192
    /*
    bm_image tmp_u8_rgb;
    if (bm_image_create(handle_, S, S, FORMAT_RGB_PLANAR, dtype_u8, &tmp_u8_rgb) != BM_SUCCESS) {
        printf("[Face3D] bm_image_create tmp_u8_rgb failed\n");
        bm_image_destroy(src_u8_rgb);
        return false;
    }
    if (bm_image_alloc_dev_mem(tmp_u8_rgb, BMCV_HEAP1_ID) != BM_SUCCESS) {
        printf("[Face3D] alloc tmp_u8_rgb dev mem failed\n");
        bm_image_destroy(src_u8_rgb);
        bm_image_destroy(tmp_u8_rgb);
        return false;
    }
    */

    // 7) 初始化输出图
    //if (bm_image_create(handle_, S, S, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &aligned_rgb_f32) != BM_SUCCESS) {
    if (bm_image_create(handle_, S, S, src_u8_rgb.image_format, dtype_u8, &aligned_rgb_f32) != BM_SUCCESS) {
        printf("[Face3D] bm_image_create aligned_rgb_f32 failed\n");
        bm_image_destroy(src_u8_rgb);
        //bm_image_destroy(tmp_u8_rgb);
        return false;
    }
    if (bm_image_alloc_dev_mem(aligned_rgb_f32, BMCV_HEAP1_ID) != BM_SUCCESS) {
        printf("[Face3D] alloc aligned_rgb_f32 dev mem failed\n");
        bm_image_destroy(src_u8_rgb);
        //bm_image_destroy(tmp_u8_rgb);
        bm_image_destroy(aligned_rgb_f32);
        return false;
    }


    // 5) 组装 affine 矩阵（BMCV 的新版接口）
    bmcv_affine_image_matrix mat_img {};
    mat_img.matrix_num = 1;
    bmcv_affine_matrix mat {};
    mat.m[0] = IM.a00;
    mat.m[1] = IM.a01;
    mat.m[2] = IM.a02;
    mat.m[3] = IM.a10;
    mat.m[4] = IM.a11;
    mat.m[5] = IM.a12;
    mat_img.matrix = &mat;

    // 6) 做 warp（RGB_PLANAR U8 -> RGB_PLANAR U8）


    //bm_status_t rrrr =  bm_image_write_to_bmp(src_u8_rgb, "/home/linaro/src_u8_rgb.bmp");
    //printf("II [Face3D][DEBUG] bm_image_write_to_bmp src_u8_rgb ret=%d\n", rrrr);

    //ret = bmcv_image_warp_affine(handle_, /*matrix_num=*/1, &mat_img, &src_u8_rgb, &tmp_u8_rgb,
    ret = bmcv_image_warp_affine(handle_, /*matrix_num=*/1, &mat_img, &src_u8_rgb, &aligned_rgb_f32,
    //ret = bmcv_image_warp_affine(handle_, /*matrix_num=*/1, &mat_img, &src_bgr_in, &aligned_rgb_f32,
                                 /*use_bilinear=*/1);
    if (ret != BM_SUCCESS) {
        printf("[Face3D] warp_affine failed, ret=%d\n", ret);
        dump_bm_image_info("[Face3D] src_u8_rgb", src_u8_rgb);
        //dump_bm_image_info("[Face3D] tmp_u8_rgb", tmp_u8_rgb);
        bm_image_destroy(src_u8_rgb);
        bm_image_destroy(aligned_rgb_f32);
        //bm_image_destroy(tmp_u8_rgb);
        return false;
    }

    //rrrr =  bm_image_write_to_bmp(tmp_u8_rgb, "/home/linaro/tmp_u8_rgb.bmp");
    //printf("II [Face3D][DEBUG] bm_image_write_to_bmp tmp_u8_rgb ret=%d\n", rrrr);

    // 7) 再把 U8 → F32（归一化到 0..1）
/*
    if (bm_image_create(handle_, S, S, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &aligned_rgb_f32) != BM_SUCCESS) {
        printf("[Face3D] bm_image_create aligned_rgb_f32 failed\n");
        bm_image_destroy(src_u8_rgb);
        bm_image_destroy(tmp_u8_rgb);
        return false;
    }
    if (bm_image_alloc_dev_mem(aligned_rgb_f32, BMCV_HEAP1_ID) != BM_SUCCESS) {
        printf("[Face3D] alloc aligned_rgb_f32 dev mem failed\n");
        bm_image_destroy(src_u8_rgb);
        bm_image_destroy(tmp_u8_rgb);
        bm_image_destroy(aligned_rgb_f32);
        return false;
    }
*/

/*
    bmcv_convert_to_attr to_f32 {};
    // 1/1.0 的 alpha = 1.f，若需归一化 0..1，alpha = 1/255.f
    to_f32.alpha_0 = 1.f ;
    to_f32.beta_0 = 0.f;
    to_f32.alpha_1 = 1.f ;
    to_f32.beta_1 = 0.f;
    to_f32.alpha_2 = 1.f ;
    to_f32.beta_2 = 0.f;

    ret = bmcv_image_convert_to(handle_, 1, to_f32, &tmp_u8_rgb, &aligned_rgb_f32);
    if (ret != BM_SUCCESS) {
        printf("[Face3D] convert_to -> F32 failed, ret=%d\n", ret);
        bm_image_destroy(src_u8_rgb);
        bm_image_destroy(tmp_u8_rgb);
        bm_image_destroy(aligned_rgb_f32);
        return false;
    }
    */

    // 清理中间物
    bm_image_destroy(src_u8_rgb);
    //bm_image_destroy(tmp_u8_rgb);

    //dump_bm_image_info("[Face3D][DEBUG] aligned image", aligned_rgb_f32);
    //printf("[Face3D][DEBUG] M = [[%.4f %.4f %.4f],[%.4f %.4f %.4f]]\n", M.a00, M.a01, M.a02, M.a10, M.a11, M.a12);
    //printf("[Face3D][DEBUG] IM= [[%.4f %.4f %.4f],[%.4f %.4f %.4f]]\n", IM.a00, IM.a01, IM.a02, IM.a10, IM.a11, IM.a12);

    // 检查中心点映射
    //float cx_ = aligned_rgb_f32.width / 2.0f;
    //float cy_ = aligned_rgb_f32.height / 2.0f;
    //float ox_ = M.a00 * cx_ + M.a01 * cy_ + M.a02;
    //float oy_ = M.a10 * cx_ + M.a11 * cy_ + M.a12;
    //printf("[Face3D][DEBUG] center of aligned (%.1f,%.1f) -> orig (%.1f,%.1f)\n", cx_, cy_, ox_, oy_);

    return true;
}

// ========== fc1 -> 68×3 ==========
bool Face3D::fc1_to_3d68(const std::vector<float>& fc1, const Affine2x3& IM, std::array<Point3f, 68>& pts3d, std::array<Point3f, 68>& pts3d_crop) const {
    if ((int)fc1.size() != 3309){
        printf("fc1 size error: %lu\n", fc1.size());
        return false;
    }
    const int s = cfg_.input_size / 2;  // 96
    const float* p = fc1.data() + (3309 - 68 * 3);

    Point2f in2d[68];
    for (int i = 0; i < 68; i++) {
        float x = (p[i * 3 + 0] + 1.f) * s, y = (p[i * 3 + 1] + 1.f) * s, z = p[i * 3 + 2] * s;
        pts3d_crop[i] = { x, y, z };
        in2d[i] = { x, y };
        //printf("pt[%d] crop: x=%f, y=%f, z=%f\n", i, x, y, z);
    }
    printf("[Face3D][DEBUG] crop point[0]=%.1f,%.1f, z=%.1f\n", pts3d_crop[0].x, pts3d_crop[0].y, pts3d_crop[0].z);
    Point2f out2d[68];
    affineTransformPoints(IM, in2d, out2d, 68);
    for (int i = 0; i < 68; i++){
        
        pts3d[i] = { out2d[i].x, out2d[i].y, pts3d_crop[i].z };
        //printf("pt[%d] orig: x=%f, y=%f, z=%f\n", i, pts3d[i].x, pts3d[i].y, pts3d[i].z);
    }
    printf("[Face3D][DEBUG] img point[0]=%.1f,%.1f\n", pts3d[0].x, pts3d[0].y);
    return true;
}

bool Face3D::load_meanshape(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[Face3D] failed to open meanshape_txt: " << path << std::endl;
        return false;
    }

    std::vector<Point3f> pts;
    pts.reserve(68);

    for (std::string line; std::getline(fin, line);) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        Point3f p {};
        if (!(iss >> p.x >> p.y >> p.z)) {
            std::cerr << "[Face3D] parse error line: " << line << std::endl;
            return false;
        }
        pts.push_back(p);
    }

    if (pts.size() != 68) {
        std::cerr << "[Face3D] wrong meanshape size: " << pts.size() << std::endl;
        return false;
    }

    // ✅ 填充到 mean68_
    for (int i = 0; i < 68; ++i)
        mean68_[i] = pts[i];

    // 打印确认范围
    float minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
    for (auto& p : mean68_) {
        minx = std::min(minx, p.x);
        maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y);
        maxy = std::max(maxy, p.y);
    }
    std::printf("[Face3D] meanshape loaded, x:[%.1f,%.1f] y:[%.1f,%.1f]\n", minx, maxx, miny, maxy);

    return true;
}

// ========== 姿态 ==========
bool Face3D::estimate_pose(const std::array<Point3f, 68>& Ycrop, Pose3DOut& out) const {
    Vec3d X[68], Y[68];
    for (int i = 0; i < 68; i++) {
        X[i] = { mean68_[i].x, mean68_[i].y, mean68_[i].z };
        Y[i] = { Ycrop[i].x, Ycrop[i].y, Ycrop[i].z };
    }
    float s;
    Mat3d R;
    Vec3d t;
    umeyama_3d3d(X, Y, 68, s, R, t);
    out.s = s;
    int k = 0;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            out.R[k++] = (float)R.m[r][c];
    out.t3 = { (float)t.x, (float)t.y, (float)t.z };
    eulerFromR(R, out.pitch_deg, out.yaw_deg, out.roll_deg);
    return true;
}

// ========== 主流程 ==========
bool Face3D::infer_on_bbox(bm_image& src_bgr, const FaceBox& bbox, Pose3DOut& pose_out, Landmarks3DOut& lmk_out) {
    bm_image aligned;
    Affine2x3 M, IM;
    
    printf("src_bgr info : w=%d, h=%d\n", src_bgr.width, src_bgr.height);
    printf("bbox : x1=%f, y1=%f, x2=%f, y2=%f\n", bbox.x1, bbox.y1, bbox.x2, bbox.y2);
    printf("[Face3D][DEBUG] bbox center=(%.1f, %.1f), size=(%.1f,%.1f)\n", 0.5f * (bbox.x1 + bbox.x2), 0.5f * (bbox.y1 + bbox.y2), bbox.x2 - bbox.x1,
           bbox.y2 - bbox.y1);

    //bm_status_t rrrr =  bm_image_write_to_bmp(src_bgr, "/home/linaro/src_debug.bmp");
    //printf("[Face3D][DEBUG] bm_image_write_to_bmp src_bgr ret=%d\n", rrrr);
    if (!align_to_192_bm(src_bgr, bbox, aligned, M, IM)){
        printf("align_to_192_bm error\n");
        return false;
    }else{
        printf("align_to_192_bm success\n");
    }

    std::vector<float> fc1;

    printf("aligned image: w=%d, h=%d\n", aligned.width, aligned.height);
    printf("aligned image format=%d, dtype=%d\n", aligned.image_format, aligned.data_type);
    
    //rrrr =  bm_image_write_to_bmp(aligned, "/home/linaro/aligned_debug.bmp");
    //printf("[Face3D][DEBUG] bm_image_write_to_bmp ret=%d\n", rrrr);

    std::vector<bm_image> imgs = { aligned };

    int ret = this->pre_process(imgs);
    
    if (ret == 0) {
        printf("pre_process success\n");
        fc1 = this->InferenceOnce();
        printf("InferenceOnce success\n");
        printf("fc1 size: %lu\n", fc1.size());
    } else {
        printf("pre_process error  \n");
        return false;
    }


    /*
    bool ok = forward_fc1(aligned, fc1);
    bm_image_destroy(aligned);
    if (!ok)
        return false;
    */

    // 你已有 fc1、以及对齐时的临时裁剪图 crop_u8_bgr（BGR U8，192x192）
    //debug_dump_fc1_and_crop(fc1, aligned, M, IM);

    std::array<Point3f, 68> pts3d, pts3d_crop;
    if (!fc1_to_3d68(fc1, IM, pts3d, pts3d_crop))
        return false;

    if (!estimate_pose(pts3d_crop, pose_out))
        return false;

    //if (lmk_out) {
    lmk_out.pts3d = pts3d;
    lmk_out.pts3d_crop = pts3d_crop;
    lmk_out.M = M;
    lmk_out.IM = IM;
    //}
    return true;
}

bool Face3D::infer(bm_image& src_bgr, const std::vector<FaceBox>& dets, Pose3DOut& pose_out, Landmarks3DOut& lmk_out) {
    if (dets.empty()){
        printf("Face3D dets is empty\n");
        return false;
    }
    int idx = -1;
    float maxA = 0.f;
    for (int i = 0; i < (int)dets.size(); ++i) {
        if (dets[i].score < 0.5f)
            continue;
        float a = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
        if (a > maxA) {
            maxA = a;
            idx = i;
        }
    }
    if (idx < 0){
        printf("Face3D no valid det\n");
        return false;
    }
    return infer_on_bbox(src_bgr, dets[idx], pose_out, lmk_out);
}

int Face3D::pre_process(std::vector<bm_image>& images) {
  std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
  int image_n = images.size();

  printf("Face3D::pre_process image_n=%d\n", image_n);
  printf("width=%d, height=%d\n", images[0].width, images[0].height);

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
    //auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, images[i],
    //                                  &m_resized_imgs[i]);
#endif
    //assert(BM_SUCCESS == ret);

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
                              //m_resized_imgs.data(), m_converto_imgs.data());
                              images.data(), m_converto_imgs.data());
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

std::vector<float> Face3D::InferenceOnce(){
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
