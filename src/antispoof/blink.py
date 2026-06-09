#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
blink.py
--------
眨眼偵測模組：使用 MediaPipe Face Mesh 計算 EAR（Eye Aspect Ratio）。
作為 MiniFASNet 的輔助防偽手段，要求使用者在授權前完成眨眼動作。

EAR 公式：
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    EAR < threshold → 眼睛閉合（眨眼）

MediaPipe 左眼輪廓點索引：
    p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
MediaPipe 右眼輪廓點索引：
    p1=362, p2=385, p3=387, p4=263, p5=373, p6=380

Usage:
    from src.antispoof.blink import BlinkDetector
    detector = BlinkDetector()
    result = detector.update(frame)
    if result.blink_confirmed:
        print("眨眼確認！")
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp


# ── MediaPipe 眼睛輪廓索引 ────────────────────────────────────────────────────
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class BlinkResult:
    """眨眼偵測結果。"""

    ear: float  # 當前 EAR 值
    is_closed: bool  # 當前幀眼睛是否閉合
    blink_count: int  # 累計眨眼次數
    blink_confirmed: bool  # 是否達到要求的眨眼次數
    landmarks: Optional[object] = field(default=None, repr=False)


class BlinkDetector:
    """
    MediaPipe Face Mesh EAR 眨眼偵測器。

    Args:
        ear_threshold:      EAR 低於此值視為眼睛閉合（預設 0.25）
        required_blinks:    授權所需眨眼次數（預設 2）
        consec_frames:      連續幾幀 EAR < threshold 才算一次眨眼（預設 2）
        reset_timeout:      超過此秒數無眨眼則重置計數（預設 10 秒）
    """

    def __init__(
        self,
        ear_threshold: float = 0.25,
        required_blinks: int = 2,
        consec_frames: int = 2,
        reset_timeout: float = 10.0,
    ):
        self.ear_threshold = ear_threshold
        self.required_blinks = required_blinks
        self.consec_frames = consec_frames
        self.reset_timeout = reset_timeout

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._blink_count = 0
        self._consec_count = 0  # 連續閉眼幀數
        self._eye_closed = False  # 上一幀眼睛狀態
        self._last_blink_t = time.time()

        print(
            f"[BlinkDetector] 初始化，需要 {required_blinks} 次眨眼，EAR 門檻={ear_threshold}"
        )

    # ── EAR 計算 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _eye_aspect_ratio(landmarks, eye_indices: list, w: int, h: int) -> float:
        """
        計算單眼 EAR。

        Args:
            landmarks:   MediaPipe face landmarks
            eye_indices: 6 個眼睛輪廓點索引
            w, h:        影像寬高

        Returns:
            EAR float
        """
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_indices])

        # 垂直距離
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        # 水平距離
        C = np.linalg.norm(pts[0] - pts[3])

        ear = (A + B) / (2.0 * C + 1e-6)
        return float(ear)

    # ── 主要更新函式 ──────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> BlinkResult:
        """
        對單幀影像進行眨眼偵測。

        Args:
            frame: BGR uint8 影像

        Returns:
            BlinkResult
        """
        # 超時重置
        if time.time() - self._last_blink_t > self.reset_timeout:
            self.reset()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return BlinkResult(
                ear=1.0,
                is_closed=False,
                blink_count=self._blink_count,
                blink_confirmed=self._blink_count >= self.required_blinks,
            )

        lm = result.multi_face_landmarks[0].landmark

        left_ear = self._eye_aspect_ratio(lm, LEFT_EYE, w, h)
        right_ear = self._eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0

        is_closed = ear < self.ear_threshold

        # 眨眼狀態機：連續 N 幀閉合 → 再睜開 = 一次眨眼
        if is_closed:
            self._consec_count += 1
        else:
            if self._consec_count >= self.consec_frames and self._eye_closed:
                self._blink_count += 1
                self._last_blink_t = time.time()
                print(f"[BlinkDetector] 眨眼 #{self._blink_count}（EAR={ear:.3f}）")
            self._consec_count = 0

        self._eye_closed = is_closed

        return BlinkResult(
            ear=round(ear, 4),
            is_closed=is_closed,
            blink_count=self._blink_count,
            blink_confirmed=self._blink_count >= self.required_blinks,
            landmarks=result.multi_face_landmarks[0],
        )

    def reset(self) -> None:
        """重置眨眼計數（每次門禁流程開始前呼叫）。"""
        self._blink_count = 0
        self._consec_count = 0
        self._eye_closed = False
        self._last_blink_t = time.time()

    def draw(self, frame: np.ndarray, result: BlinkResult) -> np.ndarray:
        """
        在影像上繪製 EAR 數值與眨眼狀態。

        Args:
            frame:  原始 BGR 影像
            result: BlinkResult

        Returns:
            繪製後的影像
        """
        h, w = frame.shape[:2]
        color = (0, 255, 0) if result.blink_confirmed else (0, 165, 255)

        cv2.putText(
            frame,
            f"EAR: {result.ear:.3f}",
            (w - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Blinks: {result.blink_count}/{self.required_blinks}",
            (w - 200, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        if result.is_closed:
            cv2.putText(
                frame,
                "CLOSED",
                (w - 200, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        return frame


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import sys

    detector = BlinkDetector(required_blinks=2)

    # 用靜態圖片測試 EAR 計算
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/enrollment/henry/0000.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] 無法讀取：{img_path}")
        sys.exit(1)

    result = detector.update(img)
    print(
        f"EAR={result.ear:.4f}  is_closed={result.is_closed}  blinks={result.blink_count}"
    )
