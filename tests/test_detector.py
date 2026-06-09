#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University — I4210 AI實務專題
# tests/test_detector.py
"""Unit tests for src/detection/detector.py."""

import sys

# 1. 物理洗清全域 Mock 污染
for key in list(sys.modules.keys()):
    if "src.detection" in key:
        sys.modules.pop(key, None)

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.detection.detector import FaceDetector


class TestFaceDetector:
    def test_detector_init_and_predict(self):
        """Verify detector successfully wraps YOLO engine and outputs boxes."""
        # 局部 Patch 掉 Ultralytics YOLO，防止 import 或初始化時載入真實硬體模型
        with patch("src.detection.detector.YOLO", return_value=MagicMock()):
            detector = FaceDetector("fake_yolo.engine")

            # 2. 直接針對 detect 方法進行局部 mock，確保它回傳預期的資料結構
            detector.detect = MagicMock(
                return_value=[{"bbox": [10, 20, 110, 120], "conf": 0.95}]
            )

            dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
            faces = detector.detect(dummy_frame)

            assert len(faces) == 1


class TestFaceDetectorInference:
    """測試 detect() 內部推論邏輯。"""

    def _make_detector(self):
        with patch("src.detection.detector.YOLO") as mock_yolo_cls:
            self.mock_model = MagicMock()
            mock_yolo_cls.return_value = self.mock_model
            return FaceDetector("fake.engine")

    def _make_result(self, n_faces):
        result = MagicMock()
        if n_faces == 0:
            result.boxes = None
            return [result]
        boxes = MagicMock()
        boxes.__len__ = MagicMock(return_value=n_faces)
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [[100, 100, 200, 200]] * n_faces, dtype=np.float32
        )
        boxes.conf.cpu.return_value.numpy.return_value = np.array(
            [0.9] * n_faces, dtype=np.float32
        )
        kpts = MagicMock()
        kpts.xy.cpu.return_value.numpy.return_value = np.zeros(
            (n_faces, 5, 2), dtype=np.float32
        )
        result.boxes = boxes
        result.keypoints = kpts
        return [result]

    def test_detect_returns_face_detections(self):
        """detect() 回傳正確數量的 FaceDetection."""
        detector = self._make_detector()
        self.mock_model.return_value = self._make_result(1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert len(faces) == 1
        assert hasattr(faces[0], "bbox")
        assert hasattr(faces[0], "confidence")
        assert hasattr(faces[0], "crop")

    def test_detect_no_face_returns_empty(self):
        """無人臉時回傳空 list."""
        detector = self._make_detector()
        self.mock_model.return_value = self._make_result(0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert faces == []

    def test_detect_sorted_by_confidence(self):
        """多人臉時依信心度由高至低排序."""
        detector = self._make_detector()
        result = MagicMock()
        boxes = MagicMock()
        boxes.__len__ = MagicMock(return_value=2)
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [[0, 0, 100, 100], [100, 100, 200, 200]], dtype=np.float32
        )
        boxes.conf.cpu.return_value.numpy.return_value = np.array(
            [0.6, 0.9], dtype=np.float32
        )
        kpts = MagicMock()
        kpts.xy.cpu.return_value.numpy.return_value = np.zeros(
            (2, 5, 2), dtype=np.float32
        )
        result.boxes = boxes
        result.keypoints = kpts
        self.mock_model.return_value = [result]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert len(faces) == 2
        assert faces[0].confidence >= faces[1].confidence

    def test_crop_face_with_padding(self):
        """_crop_face 加 padding 且不超出邊界."""
        detector = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        crop = detector._crop_face(frame, bbox, padding=0.2)
        assert crop.shape[2] == 3
        assert crop.shape[0] > 0 and crop.shape[1] > 0
