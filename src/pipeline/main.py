#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
main.py
-------
智慧門禁系統主程式：整合 YOLOv8n-face + MobileFaceNet + MiniFASNet
透過 IMX219 CSI 相機進行即時人臉辨識與活體偵測。

決策邏輯（三條件同時滿足才授權）：
  1. cosine similarity >= 0.85
  2. 連續 3 幀匹配同一人
  3. 活體偵測通過（is_live = True）

Usage:
    pdm run python src/pipeline/main.py
    pdm run python src/pipeline/main.py --no-display   # headless 模式
"""

import argparse

# 加入專案根目錄至 path
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.antispoof.antispoof import AntiSpoof
from src.detection.detector import FaceDetector
from src.recognition.recognizer import FaceRecognizer


# ── Config ───────────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── GStreamer pipeline ────────────────────────────────────────────────────────
def gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 20,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


# ── Decision buffer ───────────────────────────────────────────────────────────
class TemporalVoter:
    """
    時間一致性投票器：連續 N 幀匹配同一人才授權。
    防止單幀照片攻擊。
    """

    def __init__(self, required_frames: int = 3):
        self.required = required_frames
        self.buffer: deque = deque(maxlen=required_frames)

    def vote(self, name: str) -> tuple[bool, str]:
        """
        加入一幀的辨識結果，回傳是否達成授權條件。

        Returns:
            (authorized, name) — authorized=True 表示連續 N 幀同一人
        """
        self.buffer.append(name)
        if len(self.buffer) == self.required:
            names = list(self.buffer)
            if len(set(names)) == 1 and names[0] != "unknown":
                return True, names[0]
        return False, "unknown"

    def reset(self) -> None:
        self.buffer.clear()


# ── Access decision ───────────────────────────────────────────────────────────
def make_decision(
    recog_result,
    liveness_result,
    voter: TemporalVoter,
) -> tuple[bool, str, str]:
    """
    綜合辨識、活體、時間一致性，輸出最終決策。

    Returns:
        (granted, name, reason)
    """
    # 條件 1：活體偵測
    if not liveness_result.is_live:
        voter.reset()
        return False, "unknown", f"spoof detected (score={liveness_result.score:.2f})"

    # 條件 2：相似度門檻
    if not recog_result.authorized:
        voter.reset()
        return False, "unknown", f"similarity too low ({recog_result.similarity:.3f})"

    # 條件 3：時間一致性（連續 3 幀）
    authorized, name = voter.vote(recog_result.name)
    if authorized:
        return (
            True,
            name,
            f"similarity={recog_result.similarity:.3f}, liveness={liveness_result.score:.3f}",
        )

    return False, recog_result.name, f"waiting frames ({len(voter.buffer)}/{voter.required})"


# ── Draw overlay ──────────────────────────────────────────────────────────────
def draw_overlay(
    frame: np.ndarray,
    faces,
    decisions: list,
    fps: float,
) -> np.ndarray:
    """在影像上繪製偵測結果與系統狀態。"""
    vis = frame.copy()
    h, w = vis.shape[:2]

    for i, (face, (granted, name, reason)) in enumerate(zip(faces, decisions)):
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = (0, 255, 0) if granted else (0, 0, 255)

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 標籤
        label = f"{name} ({face.confidence:.2f})"
        cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 5 keypoints
        for kp in face.keypoints:
            kx, ky = int(kp[0]), int(kp[1])
            cv2.circle(vis, (kx, ky), 3, (0, 255, 255), -1)

        # 狀態訊息
        status = "ACCESS GRANTED" if granted else reason
        cv2.putText(vis, status, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # FPS
    cv2.putText(vis, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    return vis


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(config: dict, display: bool = True) -> None:
    """
    主要 pipeline 迴圈：Sense → Process → Decide → (Act)

    Args:
        config:  config.yaml 內容
        display: 是否顯示即時畫面（headless 模式設為 False）
    """
    cfg_models = config["models"]
    cfg_recog = config["recognition"]

    # 載入三個模型
    detector = FaceDetector(
        engine_path=cfg_models["yolo"]["engine"],
        conf_threshold=0.5,
        input_size=cfg_models["yolo"]["input_size"],
    )
    recognizer = FaceRecognizer(
        engine_path=cfg_models["mobilefacenet"]["engine"],
        db_path=cfg_recog["db_path"],
        threshold=cfg_recog["similarity_threshold"],
    )
    antispoof = AntiSpoof(
        engine_path=cfg_models["minifasnet"]["engine"],
    )

    voter = TemporalVoter(required_frames=cfg_recog["confirm_frames"])

    # 開啟相機
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("無法開啟 CSI 相機")

    print("\n[Pipeline] 啟動！按 Q 或 Ctrl+C 結束。\n")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # ── Sense：人臉偵測 ───────────────────────────────────────────
            faces = detector.detect(frame)

            decisions = []
            for face in faces:
                if face.crop is None or face.crop.size == 0:
                    decisions.append((False, "unknown", "no crop"))
                    continue

                # ── Process：辨識 + 活體 ─────────────────────────────────
                recog = recognizer.match(face.crop)
                liveness = antispoof.predict(face.crop)

                # ── Decide：三條件判斷 ───────────────────────────────────
                granted, name, reason = make_decision(recog, liveness, voter)
                decisions.append((granted, name, reason))

                # Log
                print(
                    f"[Frame {frame_count:05d}] "
                    f"name={recog.name:<10} sim={recog.similarity:.3f} "
                    f"live={liveness.score:.3f} "
                    f"{'✅ GRANTED' if granted else '❌ ' + reason}"
                )

                # ── Act：授權觸發（Week 13 GPIO 整合） ───────────────────
                if granted:
                    voter.reset()  # 重置，避免重複觸發

            # ── FPS 計算 ──────────────────────────────────────────────────
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            # ── Display ───────────────────────────────────────────────────
            if display:
                vis = draw_overlay(frame, faces, decisions, fps)
                cv2.imshow("Smart Access Control", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n[Pipeline] 使用者中斷。")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        print("[Pipeline] 結束。")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="智慧門禁系統主程式")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="config.yaml 路徑",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="headless 模式（不顯示畫面）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, display=not args.no_display)


if __name__ == "__main__":
    main()
