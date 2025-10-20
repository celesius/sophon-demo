#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrfd_kps_new_opencv_arc.py
外挂 ArcFace：在已验证可用的 scrfd_kps_new_opencv.py 检测脚本上增加人脸识别功能
"""
import os
import cv2
import time
import glob
import argparse
import logging
import numpy as np
import importlib.util
from sophon import sail

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ================== 动态加载检测模块 ==================
def load_detector_module():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    core_path = os.path.join(base_dir, "scrfd_kps_new_opencv.py")
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"未找到检测脚本：{core_path}")
    spec = importlib.util.spec_from_file_location("scrfd_kps_core", core_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "SCRFD"):
        raise AttributeError("scrfd_kps_new_opencv.py 中未找到 SCRFD 类")
    return mod


# ================== ArcFace 五点对齐 ==================
ARC_SRC_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_by_5pts(bgr_img, kps_5x2, out_size=(112, 112)):
    """使用5点关键点做仿射对齐"""
    if bgr_img is None or kps_5x2 is None or len(kps_5x2) != 5:
        return None
    M, _ = cv2.estimateAffinePartial2D(kps_5x2.astype(np.float32), ARC_SRC_5PTS, method=cv2.LMEDS)
    if M is None:
        return None
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    aligned = cv2.warpAffine(rgb, M, out_size)
    return aligned


# ================== ArcFace 模型封装 ==================
class ArcFaceBM:
    def __init__(self, model_path, dev_id=0):
        self.model_path = model_path
        self.dev_id = dev_id

        # 枚举 IOMode
        mode_candidates = [attr for attr in dir(sail.IOMode) if not attr.startswith("_")]
        print(f"[DEBUG] 可用的 sail.IOMode 枚举: {mode_candidates}")
        if hasattr(sail.IOMode, "SYSIO"):
            selected_mode = sail.IOMode.SYSIO
        else:
            selected_mode = getattr(sail.IOMode, mode_candidates[0])
        print(f"[INFO] 使用 IOMode.{selected_mode.name}")

        self.engine = sail.Engine(model_path, dev_id, selected_mode)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]
        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)

        print(f"[INFO] ArcFace 模型加载成功: {model_path}")
        print(f"[DEBUG] ArcFace 输入形状: {self.input_shape}")

    def preprocess(self, img):
        """单张 RGB/BGR 图像 → NCHW[-1,1]"""
        if img is None or not hasattr(img, "shape") or img.size == 0:
            print("[WARN] preprocess received empty img, skip.")
            return None
        if img.ndim != 3 or img.shape[2] != 3:
            print(f"[WARN] preprocess expects HWC 3ch image, got shape={img.shape}, skip.")
            return None
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def forward(self, img):
        inp = self.preprocess(img)
        if inp is None:
            return None
        outputs = self.engine.process(self.graph_name, {self.input_name: inp})
        feat = outputs[self.output_name].astype(np.float32)
        if feat.ndim == 1:
            feat = feat[None, :]
        feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
        return feat

    def __call__(self, img):
        return self.forward(img)

# ================== 构建人脸图库 ==================
def build_gallery(detector, arc_engine, gallery_dir, det_conf=0.5):
    feats, names = [], []
    all_images = []
    for root, _, files in os.walk(gallery_dir):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                all_images.append(os.path.join(root, f))
    if not all_images:
        print(f"[WARN] gallery {gallery_dir} is empty")
        return np.empty((0,512),dtype=np.float32), []

    print(f"[INFO] Building gallery from: {gallery_dir}")
    for fp in all_images:
        name = os.path.basename(os.path.dirname(fp))
        img = cv2.imdecode(np.fromfile(fp,dtype=np.uint8),-1)
        if img is None: continue
        if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        if img.shape[2]==4: img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)

        dets = detector([img])[0]
        if dets.shape[0]==0:
            print(f"[WARN] No face detected in {fp}")
            continue
        det = dets[np.argmax(dets[:,14])]
        if det[14]<det_conf:
            continue
        kps = det[4:14].reshape(5,2)
        aligned = align_by_5pts(img,kps)
        if aligned is None: continue
        feat = arc_engine(aligned)
        if feat is None or feat.size==0: continue
        feats.append(feat[0])
        names.append(name)
        print(f"[INFO] 已录入 {name} : {os.path.basename(fp)}")

    if not feats:
        return np.empty((0,512),dtype=np.float32), []
    feats = np.stack(feats,axis=0)
    print(f"[INFO] 图库构建完成，共 {len(names)} 张, 维度={feats.shape}")
    return feats, names


# ================== 绘制辅助 ==================
def draw_with_names_kps(img, det, names=None, sims=None):
    out = img.copy()
    if det is None or det.shape[0]==0: return out
    if names is None: names=[""]*det.shape[0]
    if sims is None: sims=[None]*det.shape[0]
    for i,row in enumerate(det):
        x1,y1,x2,y2=row[:4].astype(np.int32)
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,255),2)
        text = names[i]
        if sims[i] is not None: text += f" {sims[i]:.2f}"
        cv2.putText(out,text,(x1,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        if det.shape[1]>=16:
            kps=row[4:14].reshape(5,2).astype(np.int32)
            for (px,py) in kps:
                cv2.circle(out,(px,py),2,(0,255,0),-1)
    return out


# ================== 主推理（逐张特征） ==================
def run_on_images(detector, arc_engine, gallery_feats, gallery_names, in_dir, out_dir, sim_thresh):
    os.makedirs(out_dir, exist_ok=True)
    img_files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        img_files.extend(glob.glob(os.path.join(in_dir, ext)))
    img_files.sort()

    for fp in img_files:
        img = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), -1)
        if img is None:
            continue
        dets = detector([img])[0]
        if dets.shape[0]==0:
            continue

        names, sims = [], []
        face_feats = []
        valid_boxes = []

        for det in dets:
            x1,y1,x2,y2 = det[:4].astype(np.int32)
            kps = det[4:14].reshape(5,2)
            aligned = align_by_5pts(img,kps)
            if aligned is None: continue
            feat = arc_engine(aligned)
            if feat is None: continue
            face_feats.append(feat[0])
            valid_boxes.append((x1,y1,x2,y2))

        if not face_feats:
            vis = draw_with_names_kps(img, dets)
        else:
            feats = np.stack(face_feats,axis=0)
            S = feats @ gallery_feats.T
            for i in range(feats.shape[0]):
                k = int(np.argmax(S[i]))
                sim = float(S[i,k])
                sims.append(sim)
                names.append(gallery_names[k] if sim>=sim_thresh else "Unknown")
            vis = draw_with_names_kps(img, dets[:len(names)], names, sims)

        save_path = os.path.join(out_dir, os.path.basename(fp))
        cv2.imwrite(save_path, vis)
        logging.info(f"saved -> {save_path}")


# ================== 主函数 ==================
def main(args):
    det_mod = load_detector_module()
    SCRFD = det_mod.SCRFD
    detector = SCRFD(args)
    logging.info(f"load {args.bmodel} success!")

    arc_engine, gallery_feats, gallery_names = None, None, None
    if args.arc_bmodel:
        arc_engine = ArcFaceBM(args.arc_bmodel, args.dev_id)
        if args.gallery:
            gallery_feats, gallery_names = build_gallery(detector, arc_engine, args.gallery, det_conf=args.conf_thresh)

    os.makedirs("results", exist_ok=True)
    if os.path.isdir(args.input):
        out_dir = os.path.join("results", "images_arc")
        run_on_images(detector, arc_engine, gallery_feats, gallery_names,
                      args.input, out_dir, args.sim_thresh)
    else:
        print("[WARN] 当前仅实现目录输入，如需视频/单图可扩展调用 run_on_video()。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bmodel", type=str, required=True)
    parser.add_argument("--dev_id", type=int, default=0)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--arc_bmodel", type=str, default=None)
    parser.add_argument("--gallery", type=str, default=None)
    parser.add_argument("--sim_thresh", type=float, default=0.38)
    args = parser.parse_args()
    main(args)