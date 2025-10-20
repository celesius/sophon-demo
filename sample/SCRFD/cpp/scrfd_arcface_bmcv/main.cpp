//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>

#include "ff_decode.hpp"
#include "scrfd.hpp"
#include "arcface.hpp"
using namespace std;
#define WITH_ENCODE 1

#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

// 判断文件是否存在
static bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// 自动生成不重复的 debug 文件名
static std::string next_debug_filename(const std::string& prefix = "debug_frame_", const std::string& ext = ".jpg") {
    int idx = 0;
    std::string filename;
    do {
        filename = prefix + std::to_string(idx) + ext;
        idx++;
    } while (file_exists(filename));
    return filename;
}

// 打印 + 保存 mat_bgr
static void debug_save_mat(const cv::Mat& mat_bgr, const std::string& tag = "") {
    if (mat_bgr.empty()) {
        std::cerr << "[ERROR] mat_bgr is empty!" << std::endl;
        return;
    }

    std::cout << "\n==== [DEBUG] Mat Info " << tag << " ====" << std::endl;
    std::cout << "Size: " << mat_bgr.cols << "x" << mat_bgr.rows << "  Channels: " << mat_bgr.channels() << "  Depth: " << mat_bgr.depth() << std::endl;

    if (mat_bgr.channels() == 3) {
        cv::Vec3b center = mat_bgr.at<cv::Vec3b>(mat_bgr.rows / 2, mat_bgr.cols / 2);
        std::cout << "Center pixel (B,G,R) = [" << std::setw(3) << (int)center[0] << ", " << std::setw(3) << (int)center[1] << ", " << std::setw(3)
                  << (int)center[2] << "]" << std::endl;
    }

    // 自动生成文件名
    std::string save_path = next_debug_filename();
    if (cv::imwrite(save_path, mat_bgr)) {
        std::cout << "[INFO] Saved debug image -> " << save_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save image -> " << save_path << std::endl;
    }
}

/**
 * arcface 推理测试程序，输入图像地址，输出特征向量
  1. 输入图像必须是 BGR 格式
  2. 图像中必须已经是人脸区域（不含背景），且人脸尽量居中
  3. 图像尺寸必须是 112x112
  4. 输出特征向量维度是 512，float 类型
 */
void test_arcface_inference(const std::string& img_path, ArcFaceBM& arcface, std::vector<float>& feat) {
    cv::Mat bgr = cv::imread(img_path);
    if (bgr.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << img_path << std::endl;
        return;
    }
    if (bgr.cols != 112 || bgr.rows != 112) {
        std::cerr << "[ERROR] Image size must be 112x112, but got " << bgr.cols << "x" << bgr.rows << std::endl;
        return;
    }

    feat.clear();
    feat = arcface.Forward112BGR(bgr);
    if (feat.size() != 512) {
        std::cerr << "[ERROR] Expected feature size 512, but got " << feat.size() << std::endl;
        return;
    }

    std::cout << "Feature vector (first 10 values): ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << feat[i] << " ";
    }
    std::cout << "..." << std::endl;
}

/**
 * 测试arcface推理两个图片，推理结果进行余弦函数对比输出对比结果
  1. 输入图像必须是 BGR 格式
  2. 图像中必须已经是人脸区域（不含背景），且人脸尽量居中
  3. 图像尺寸必须是 112x112
  4. 输出特征向量维度是 512，float 类型
  5. 输出余弦相似度，[-1,1]，越大表示越相似 
*/

void test_arcface_inferece_2_pic(const std::string& img_path1, const std::string& img_path2, ArcFaceBM& arcface) {
    std::vector<float> feat1, feat2;
    test_arcface_inference(img_path1, arcface, feat1);
    test_arcface_inference(img_path2, arcface, feat2);
    if (feat1.size() == 512 && feat2.size() == 512) {
        float sim = ArcFaceBM::CosineSim(feat1, feat2);
        std::cout << "Cosine similarity between the two images: " << sim << std::endl;
    } else {
        std::cerr << "[ERROR] Feature extraction failed for one or both images." <<  std::endl;
    }

}


int main(int argc, char* argv[]) {
  cout.setf(ios::fixed);
  // get params
  const char* keys =
      "{bmodel | ../../models/BM1684/scrfd_10g_kps_fp32_1b.bmodel | bmodel "
      "file path}"
      "{dev_id | 0 | TPU device id}"
      "{conf_thresh | 0.001 | confidence threshold for filter boxes}"
      "{nms_thresh | 0.6 | iou threshold for nms}"
      "{help | 0 | print help information.}"
      "{eval | False | if true then gen result_txt}"
      "{input | ../../datasets/test | input path, images direction or video "
      "file path}"
      "{arc_bmodel | ../../models/BM1684/w600k_r50_f16_bm1688.bmodel | arcface bmodel file path}"
      "{gallery | ../../datasets/gallery | face feature gallery dir}"
      "{sim_thresh | 0.3 | face feature similarity threshold, [0.0, 1.0], the larger the more strict}"
      "{arctest_1 | ./test_data/arcface_test1.jpg | arcface test image 1}"
      "{arctest_2 | ./test_data/arcface_test2.jpg | arcface test image 2}"
      "{is_arctest | False | if true then test arcface inference for 2 images}"
      
      ;
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  string bmodel_file = parser.get<string>("bmodel");
  string input = parser.get<string>("input");
  int dev_id = parser.get<int>("dev_id");
  bool eval = parser.get<bool>("eval");
  
  string arcface_bmodel_file = parser.get<string>("arc_bmodel");
  std::string gallery_dir = parser.get<string>("gallery");
  float sim_thresh = parser.get<float>("sim_thresh");

  string arcface_test1_path = parser.get<string>("arctest_1");
  string arcface_test2_path = parser.get<string>("arctest_2");
  bool is_arctest = parser.get<bool>("is_arctest");

  // check params
  struct stat info;
  if (stat(bmodel_file.c_str(), &info) != 0) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }
  if (stat(input.c_str(), &info) != 0) {
    cout << "Cannot find input path." << endl;
    exit(1);
  }

  // creat handle
  BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
  cout << "set device id: " << dev_id << endl;
  bm_handle_t h = handle->handle();

  // load bmodel
  shared_ptr<BMNNContext> bm_ctx =
      make_shared<BMNNContext>(handle, bmodel_file.c_str());

  shared_ptr<BMNNContext> arcface_bm_ctx =
      make_shared<BMNNContext>(handle, arcface_bmodel_file.c_str());

  printf("load bmodel %s success!\n", bmodel_file.c_str());
  printf("load arcface bmodel %s success!\n", arcface_bmodel_file.c_str());

  // initialize net
  Scrfd scrfd(bm_ctx);
  CV_Assert(0 == scrfd.Init(parser.get<float>("conf_thresh"),
                            parser.get<float>("nms_thresh")));

  ArcFaceBM arcface(arcface_bm_ctx);
  CV_Assert(0 == arcface.Init(0.38));


  // profiling
  TimeStamp scrfd_ts;
  TimeStamp* ts = &scrfd_ts;
  scrfd.enableProfile(&scrfd_ts);

  // get batch_size
  int batch_size = scrfd.batch_size();

  // creat save path
  if (access("results", 0) != F_OK) mkdir("results", S_IRWXU);
  if (access("results/images", 0) != F_OK) mkdir("results/images", S_IRWXU);
  const string prediction_dir = "../../tools/prediction_dir";
  const string result_dir = "./results/txt_results";
  if (access("./results/txt_results", 0) != F_OK)
    mkdir("./results/txt_results", S_IRWXU);
  if (eval) {
    if (access("../../tools/prediction_dir", 0) != F_OK)
      mkdir("../../tools/prediction_dir", S_IRWXU);
  }

  if (is_arctest) {
      std::cout << "=== ArcFace Inference Test for 2 Images ===" << std::endl;
      test_arcface_inferece_2_pic(arcface_test1_path, arcface_test2_path, arcface);
      return 0;
  }

  bool build_ok = arcface.BuildGallery(gallery_dir, h, scrfd, 0.5);
  if (!build_ok) {
      std::cerr << "Failed to build gallery from: " << gallery_dir << std::endl;
      //return -1;
  }else {
      std::cout << "Built gallery with " << arcface.gallery_.size() << " identities from: " << gallery_dir << std::endl;
  }

  // test images
  if (info.st_mode & S_IFDIR) {
    // get files
    vector<string> files_vector;
    scrfd.readDirectory(input, files_vector, true);
    std::sort(files_vector.begin(), files_vector.end());

    vector<bm_image> batch_imgs;
    vector<string> batch_names;
    vector<ScrfdBoxVec> batch_boxes;
    vector<cvai_face_info_t> boxes;
    int cn = files_vector.size();
    int id = 0;
    for (vector<string>::iterator iter = files_vector.begin();
         iter != files_vector.end(); iter++) {
      string img_file = *iter;
      id++;
      size_t last_slash_index = img_file.find_last_of("/\\");
      size_t second_last_slash_index =
          img_file.find_last_of("/\\", last_slash_index - 1);
      std::string directory_name =
          img_file.substr(second_last_slash_index + 1,
                          last_slash_index - second_last_slash_index - 1);
      std::cout << id << "/" << cn << ", img_file: " << img_file << std::endl;
      ts->save("decode time");
      bm_image bmimg;
      picDec(h, img_file.c_str(), bmimg);
      ts->save("decode time");
      size_t index = img_file.rfind("/");
      string img_name = img_file.substr(index + 1);
      batch_imgs.push_back(bmimg);
      batch_names.push_back(img_name);

      iter++;
      bool end_flag = (iter == files_vector.end());
      iter--;
      if ((batch_imgs.size() == batch_size || end_flag) &&
          !batch_imgs.empty()) {
        // predict
        CV_Assert(0 == scrfd.Detect(batch_imgs, batch_boxes));
        for (int i = 0; i < batch_imgs.size(); i++) {
          bm_image frame;
#if WITH_ENCODE
          if (batch_imgs[i].image_format != 0) {
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width,
                            FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }
          printf("batch_imgs size: %d\n", batch_imgs.size());
#endif
          cv::Mat mat_bgr;
          cv::bmcv::toMAT(&frame, mat_bgr);
          std::cout << "==== [DEBUG] Mat Info ====" << std::endl;
          std::cout << "Size: " << mat_bgr.cols << "x" << mat_bgr.rows << std::endl;
          std::cout << "Channels: " << mat_bgr.channels() << std::endl;
          std::cout << "Depth: " << mat_bgr.depth() << " ("
                    << (mat_bgr.depth() == CV_8U        ? "CV_8U"
                            : mat_bgr.depth() == CV_32F ? "CV_32F"
                                                        : std::to_string(mat_bgr.depth()))
                    << ")" << std::endl;
          std::cout << "Type: " << mat_bgr.type() << std::endl;
          std::string save_path = "debug_frame_" + std::to_string(i) + ".jpg";
          bool ok = cv::imwrite(save_path, mat_bgr);
          if (!ok)
              std::cerr << "[ERROR] Failed to save image: " << save_path << std::endl;
          else
              std::cout << "[INFO] Saved image: " << save_path << std::endl;
          //debug_save_mat(mat_bgr, "./frame_" + std::to_string(i));  //image 正确

          if (eval) {
            string base_name =
                batch_names[i].substr(0, batch_names[i].rfind('.'));
            string txt_name = base_name + ".txt";
            string sub_result_directory = prediction_dir + "/" + directory_name;
            if (access(sub_result_directory.c_str(), 0) != F_OK)
              mkdir(sub_result_directory.c_str(), S_IRWXU);
            string file_path =
                prediction_dir + "/" + directory_name + "/" + txt_name;
            ofstream result_file(file_path);
            if (result_file.is_open()) {
              result_file << batch_names[i] << std::endl;
              result_file << batch_boxes[i].size() << std::endl;
              for (auto boxes : batch_boxes[i]) {
                cvai_bbox_t b_box = boxes.bbox;
                float x1 = (float)b_box.x1;
                float y1 = (float)b_box.y1;
                float x2 = (float)b_box.x2;
                float y2 = (float)b_box.y2;
                cvai_pts_t pti = boxes.pts;
                float bbox_width = std::abs(x2 - x1);
                float bbox_height = std::abs(y2 - y1);
                result_file << x1 << " " << y1 << " " << bbox_width << " "
                            << bbox_height << " " << b_box.score << endl;
              }
              result_file.close();
            } else {
            }
          } else {
            string base_name =
                batch_names[i].substr(0, batch_names[i].rfind('.'));
            string txt_name = base_name + ".txt";
            string sub_result_directory = result_dir + "/" + directory_name;
            if (access(sub_result_directory.c_str(), 0) != F_OK)
              mkdir(sub_result_directory.c_str(), S_IRWXU);
            string file_path =
                result_dir + "/" + directory_name + "/" + txt_name;
            ofstream result_file(file_path);
            if (result_file.is_open()) {
              result_file << batch_names[i] << std::endl;
              result_file << batch_boxes[i].size() << std::endl;
              for (auto boxes : batch_boxes[i]) {
                cvai_bbox_t b_box = boxes.bbox;
                float x1 = (float)b_box.x1;
                float y1 = (float)b_box.y1;
                float x2 = (float)b_box.x2;
                float y2 = (float)b_box.y2;
                cvai_pts_t pti = boxes.pts;
                /*
                std::cout << "==== [DEBUG] Mat Info ====" << std::endl;
                std::cout << "Size: " << mat_bgr.cols << "x" << mat_bgr.rows << std::endl;
                std::cout << "Channels: " << mat_bgr.channels() << std::endl;
                std::cout << "Depth: " << mat_bgr.depth() << " ("
                          << (mat_bgr.depth() == CV_8U        ? "CV_8U"
                                  : mat_bgr.depth() == CV_32F ? "CV_32F"
                                                              : std::to_string(mat_bgr.depth()))
                          << ")" << std::endl;
                std::cout << "Type: " << mat_bgr.type() << std::endl;
                std::string save_path = "debug_frame_" + std::to_string(i) + ".jpg";
                bool ok = cv::imwrite(save_path, mat_bgr);
                if (!ok)
                    std::cerr << "[ERROR] Failed to save image: " << save_path << std::endl;
                else
                    std::cout << "[INFO] Saved image: " << save_path << std::endl;
                debug_save_mat(mat_bgr, "./frame_" + std::to_string(i));  //image 正确
                */
                cv::Mat aligned_112 = ArcFaceBM::AlignBy5PtsRGB(mat_bgr, pti, 112, 112);
                if (aligned_112.empty()){
                    std::cerr << "[ERROR] aligned_112 is empty, skip draw this face." << std::endl;
                    //continue;
                }
                
                debug_save_mat(aligned_112, "./frame_" + std::to_string(i));  //image 正确

                std::vector<float> feat = arcface.Forward112RGB(aligned_112);
                std::cout << "det feat: ";
                for (size_t i = 0; i < feat.size(); i++) {
                  std::cout << feat[i] << " ";
                }
                std::cout << std::endl;
                
                std::string name;
                float sim = 0.f;
                if (arcface.MatchGallery(feat, name, sim) ){
                    if (sim >= 0.38f){
                      printf("Face recognized: %s (sim=%.4f)\n", name.c_str(), sim);
                    }else{
                      printf("Face unknown (best sim=%.4f)\n", sim);
                    }
                }
                else{
                    //cv::putText(frame_bgr, "Unknown", cv::Point(box.x1, box.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, { 0, 0, 255 }, 2);
                }

                float bbox_width = std::abs(x2 - x1);
                float bbox_height = std::abs(y2 - y1);
                result_file << x1 << " " << y1 << " " << bbox_width << " "
                            << bbox_height << " " << b_box.score << endl;
#if DEBUG
                cout << "score = " << b_box.score << " (x=" << x1 << ",y=" << y1
                     << ",w=" << bbox_width << ",h=" << bbox_height << ")"
                     << endl;
#endif
                // draw image
                scrfd.draw_bmcv(h, pti, b_box.score, int(x1), int(y1),
                                int(bbox_width), int(bbox_height),
                                batch_imgs[i], false, false);
              }
              result_file.close();
            } else {
            }
          }
          if (!eval) {
            // save image
#if WITH_ENCODE
            void* jpeg_data = NULL;
            size_t out_size = 0;
            int ret = bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data,
                                          &out_size);
            if (ret == BM_SUCCESS) {
              string img_file = "results/images/" + batch_names[i];
              FILE* fp = fopen(img_file.c_str(), "wb");
              fwrite(jpeg_data, out_size, 1, fp);
              fclose(fp);
            }
            free(jpeg_data);
#endif
          }
          bm_image_destroy(batch_imgs[i]);
        }
        batch_imgs.clear();
        batch_names.clear();
        batch_boxes.clear();
        boxes.clear();
      }
    }
    // save results
    size_t index = input.rfind("/");
    if (index == input.length() - 1) {
      input = input.substr(0, input.length() - 1);
      index = input.rfind("/");
    }
    string dataset_name = input.substr(index + 1);
    index = bmodel_file.rfind("/");
    string model_name = bmodel_file.substr(index + 1);
    string result_output_dir;
    if (eval) {
      result_output_dir = prediction_dir;
    } else {
      result_output_dir = result_dir;
    }
    cout << "================" << endl;
    cout << "result saved in " << result_output_dir << endl;
  }
  // test video
  else {
    VideoDecFFM decoder;
    decoder.openDec(&h, input.c_str());
    int id = 0;
    vector<bm_image> batch_imgs;
    vector<ScrfdBoxVec> batch_boxes;
    vector<cvai_face_info_t> boxes;
    bool end_flag = false;
    while (!end_flag) {
      bm_image* img = decoder.grab();
      if (!img) {
        end_flag = true;
      } else {
        batch_imgs.push_back(*img);
        delete img;
        img = nullptr;
      }
      if ((batch_imgs.size() == batch_size || end_flag) &&
          !batch_imgs.empty()) {
        CV_Assert(0 == scrfd.Detect(batch_imgs, batch_boxes));
        for (int i = 0; i < batch_imgs.size(); i++) {
          id++;
          cout << id << ", det_nums: " << batch_boxes[i].size() << endl;
          if (batch_imgs[i].image_format != 0) {
            bm_image frame;
            bm_image_create(h, batch_imgs[i].height, batch_imgs[i].width,
                            FORMAT_YUV420P, batch_imgs[i].data_type, &frame);
            bmcv_image_storage_convert(h, 1, &batch_imgs[i], &frame);
            bm_image_destroy(batch_imgs[i]);
            batch_imgs[i] = frame;
          }

          for (auto boxes : batch_boxes[i]) {
            cvai_bbox_t b_box = boxes.bbox;
            int x1 = (int)b_box.x1;
            int y1 = (int)b_box.y1;
            int x2 = (int)b_box.x2;
            int y2 = (int)b_box.y2;
            cvai_pts_t pti = boxes.pts;
            int bbox_width = std::abs(x2 - x1);
            int bbox_height = std::abs(y2 - y1);
#if DEBUG
            cout << "score = " << b_box.score << " (x=" << x1 << ",y=" << y1
                 << ",w=" << bbox_width << ",h=" << bbox_height << ")" << endl;
#endif
            scrfd.draw_bmcv(h, pti, b_box.score, x1, y1, bbox_width,
                            bbox_height, batch_imgs[i], false, false);
          }
          string img_file = "results/images/" + to_string(id) + ".jpg";
          void* jpeg_data = NULL;
          size_t out_size = 0;
          int ret =
              bmcv_image_jpeg_enc(h, 1, &batch_imgs[i], &jpeg_data, &out_size);
          if (ret == BM_SUCCESS) {
            FILE* fp = fopen(img_file.c_str(), "wb");
            fwrite(jpeg_data, out_size, 1, fp);
            fclose(fp);
          }
          free(jpeg_data);
          bm_image_destroy(batch_imgs[i]);
        }
        batch_imgs.clear();
        boxes.clear();
        batch_boxes.clear();
      }
    }
  }
  // print speed
  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  scrfd_ts.calbr_basetime(base_time);
  scrfd_ts.build_timeline("scrfd test");
  scrfd_ts.show_summary("scrfd test");
  scrfd_ts.clear();

  return 0;
}
