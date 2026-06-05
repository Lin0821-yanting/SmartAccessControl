#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
detector.py
-----------
YOLOv8n-face TensorRT FP16 推論器。
輸入：BGR 影像（任意大小）
輸出：List[FaceDetection]，每個包含 bbox、信心度、5 個關鍵點
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class FaceDetection:
    """單一人臉偵測結果。"""
    bbox: np.ndarray        # [x1, y1, x2, y2] float32
    confidence: float       # 信心度 0~1
    keypoints: np.ndarray   # (5, 2) float32，順序：左眼、右眼、鼻子、嘴左、嘴右
    crop: np.ndarray = field(default=None, repr=False)  # 裁切後的人臉 BGR


class FaceDetector:
    """
    YOLOv8n-face TensorRT FP16 人臉偵測器。

    Args:
        engine_path: TensorRT engine 檔案路徑
        conf_threshold: 信心度門檻（預設 0.5）
        input_size: 模型輸入大小（預設 416）
    """

    def __init__(
        self,
        engine_path: str = "models/engines/yolov8n-face-fp16.engine",
        conf_threshold: float = 0.5,
        input_size: int = 416,
    ):
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.model = YOLO(engine_path, task="pose")
        print(f"[FaceDetector] 載入 engine：{engine_path}")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        對單張影像進行人臉偵測。

        Args:
            frame: BGR uint8 影像

        Returns:
            List[FaceDetection]，依信心度由高至低排序
        """
        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()        # (N, 4)
        confs = result.boxes.conf.cpu().numpy()        # (N,)
        kpts = result.keypoints.xy.cpu().numpy()       # (N, 5, 2)

        for i in range(len(boxes)):
            bbox = boxes[i].astype(np.float32)
            conf = float(confs[i])
            keypoints = kpts[i].astype(np.float32)     # (5, 2)

            # 裁切人臉（加 20% padding）
            crop = self._crop_face(frame, bbox, padding=0.2)

            detections.append(FaceDetection(
                bbox=bbox,
                confidence=conf,
                keypoints=keypoints,
                crop=crop,
            ))

        # 依信心度由高至低排序
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _crop_face(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.2,
    ) -> np.ndarray:
        """
        裁切人臉區域，加入 padding 以包含更多臉部特徵。

        Args:
            frame:   原始影像
            bbox:    [x1, y1, x2, y2]
            padding: bbox 寬高的比例 padding

        Returns:
            裁切後的 BGR 影像
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        x1 = max(0, int(x1 - bw * padding))
        y1 = max(0, int(y1 - bh * padding))
        x2 = min(w, int(x2 + bw * padding))
        y2 = min(h, int(y2 + bh * padding))

        return frame[y1:y2, x1:x2].copy()


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/enrollment/henry/0000.jpg"
    detector = FaceDetector()

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] 無法讀取圖片：{img_path}")
        sys.exit(1)

    faces = detector.detect(img)
    print(f"偵測到 {len(faces)} 張人臉")
    for i, face in enumerate(faces):
        print(f"  [{i}] conf={face.confidence:.3f}  bbox={face.bbox}  crop={face.crop.shape}")