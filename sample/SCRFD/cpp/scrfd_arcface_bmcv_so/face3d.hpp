#pragma once
//#include "bmnnsdk.h"  // 注意：这里用你们工程里 scrfd.cpp 依赖的同一份 BMNNContext/BMNNNetwork 封装头
#include <array>
#include <memory>
#include <vector>
#include <dirent.h>
#include <algorithm>
#include "opencv2/opencv.hpp"

#include "bmnn_utils.h"
#include "bmcv_api_ext.h"
#include "bmruntime_interface.h"
#include "utils.hpp"
#include "bm_wrapper.hpp"

namespace face3d {

struct FaceBox {
    float x1, y1, x2, y2, score;
};

// 轻量数学与仿射
struct Affine2x3 {
    float a00, a01, a02, a10, a11, a12;
};
struct Point2f {
    float x, y;
};
struct Point3f {
    float x, y, z;
};
struct Mat3d {
    double m[3][3];
    static Mat3d eye() {
        Mat3d R {};
        R.m[0][0] = R.m[1][1] = R.m[2][2] = 1;
        return R;
    }
};
struct Vec3d {
    double x, y, z;
};

// 姿态输出（单位：度）；R 行主序 3x3；t3 为 3D 平移（crop 空间）
struct Pose3DOut {
    float pitch_deg { 0.f }, yaw_deg { 0.f }, roll_deg { 0.f };
    float s { 1.f };
    std::array<float, 9> R {};
    std::array<float, 3> t3 {};
};

// 可选关键点输出（SDK仅计算，ROS层去画）
struct Landmarks3DOut {
    std::array<Point3f, 68> pts3d;       // 原图坐标
    std::array<Point3f, 68> pts3d_crop;  // 192x192 裁剪坐标
    Affine2x3 M;                         // 原图->裁剪
    Affine2x3 IM;                        // 裁剪->原图
};

class Face3D {
 public:
    struct Config {
        std::string meanshape_txt;  // 68 行 "x y z"
        int input_size { 192 };     // 固定 192
        std::string input_name { "data" };
        std::string output_name { "fc1" };  // 1x3309
    };

    //static std::unique_ptr<Face3D> create_with_context(std::shared_ptr<BMNNContext> ctx3d, bm_handle_t handle);
    Face3D(std::shared_ptr<BMNNContext> context);
    ~Face3D();

    int Init(const Config& cfg);

    // 从多框选面积最大（score>=0.5），输出姿态；可选返回68×3
    bool infer(bm_image& src_bgr, const std::vector<FaceBox>& dets, Pose3DOut& pose_out, Landmarks3DOut& lmk_out);

    // 对单框
    bool infer_on_bbox(bm_image& src_bgr, const FaceBox& bbox, Pose3DOut& pose_out, Landmarks3DOut& lmk_out);

 private:
    //Face3D(std::shared_ptr<BMNNContext> ctx, bm_handle_t handle);
    // 设备侧对齐 -> RGB PLANAR F32 (1x3x192x192)
    bool align_to_192_bm(bm_image& src_bgr, const FaceBox& box, bm_image& aligned_rgb_f32, Affine2x3& M, Affine2x3& IM) const;

    // 推理（bm_image->bmrt）
    bool forward_fc1(const bm_image& aligned_rgb_f32, std::vector<float>& fc1);

    // fc1 -> 68×3；用 IM 将 2D 从裁剪映回原图
    bool fc1_to_3d68(const std::vector<float>& fc1, const Affine2x3& IM, std::array<Point3f, 68>& pts3d, std::array<Point3f, 68>& pts3d_crop) const;

    // 3D->3D 姿态（Umeyama：s,R,t3）
    bool estimate_pose(const std::array<Point3f, 68>& pts3d_crop, Pose3DOut& out) const;

    int pre_process(std::vector<bm_image>& images);

    std::vector<float> InferenceOnce();

    void debug_dump_fc1_and_crop(const std::vector<float>& fc1, const bm_image& crop_u8_bgr, const Affine2x3& M, const Affine2x3& IM);

 private:
    // ====== 无 OpenCV 的数学/工具 ======
    static void invertAffine(const Affine2x3& M, Affine2x3& IM);
    static void affineTransformPoints(const Affine2x3& M, const Point2f* in, Point2f* out, int n);
    static void eulerFromR(const Mat3d& R, float& pitch_deg, float& yaw_deg, float& roll_deg);

    static Mat3d mul(const Mat3d& A, const Mat3d& B);
    static Vec3d add(const Vec3d& a, const Vec3d& b);
    static Vec3d sub(const Vec3d& a, const Vec3d& b);
    static Vec3d smul(double s, const Vec3d& a);
    static double dot(const Vec3d& a, const Vec3d& b);
    static void jacobiEigenSymmetric3x3(const Mat3d& A, Mat3d& V, double d[3]);
    static void svd3x3_AT_A(const Mat3d& A, Mat3d& U, double s[3], Mat3d& Vt);
    static void umeyama_3d3d(const Vec3d* X, const Vec3d* Y, int n, float& s, Mat3d& R, Vec3d& t);

    bool load_meanshape(const std::string& path);

 private:
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_resized_imgs;
    std::vector<bm_image> m_converto_imgs;

    int m_net_h, m_net_w;
    int output_num;
    int max_batch;
    int min_dim;
    bmcv_convert_to_attr converto_attr;

 
private:
    bm_handle_t handle_ { nullptr };
    //const bm_network_t* net_ { nullptr };
    int in_idx_ { -1 }, out_idx_ { -1 };

    Config cfg_;
    std::array<Point3f, 68> mean68_;
};

}  // namespace face3d