#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
collect_faces.py
----------------
使用 IMX219 CSI 相機在 Jetson Orin Nano 上收集 enrollment dataset。
每位授權人員拍攝 25–30 張人臉照片，儲存至 data/enrollment/<name>/。

Usage:
    pdm run python scripts/collect_faces.py --name <person_name> --count 30
"""

import argparse
import time
from pathlib import Path

import cv2


# ── GStreamer pipeline for IMX219 ────────────────────────────────────────────
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
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )


# ── Face collector ───────────────────────────────────────────────────────────
def collect_faces(name: str, count: int, output_dir: Path) -> None:
    """
    開啟 CSI 相機並互動式收集人臉照片。

    Args:
        name:       授權人員姓名（資料夾名稱）
        count:      目標收集張數（建議 25–30）
        output_dir: enrollment 根目錄（data/enrollment/）
    """
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(save_dir.glob("*.jpg")))
    print(f"\n[INFO] 收集對象：{name}")
    print(f"[INFO] 儲存路徑：{save_dir}")
    print(f"[INFO] 已有照片：{existing} 張，目標再收集 {count} 張")
    print("\n操作說明：")
    print("  SPACE  — 拍攝一張")
    print("  A      — 自動連拍（每 0.5 秒一張）")
    print("  Q/ESC  — 結束收集\n")

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("無法開啟 CSI 相機，請確認 IMX219 連接正常且 nvarguscamerasrc 可用。")

    saved = 0
    auto_mode = False
    last_auto_time = 0.0

    while saved < count:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 讀取影格失敗，重試中...")
            continue

        # ── 顯示進度 overlay ───────────────────────────────────────────────
        display = frame.copy()
        status = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(
            display,
            f"{name}  {saved}/{count}  [{status}]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            "SPACE=拍照  A=自動  Q=結束",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )
        cv2.imshow("Enrollment - 請將臉部置於框內", display)

        key = cv2.waitKey(1) & 0xFF

        # ── 按鍵處理 ──────────────────────────────────────────────────────
        if key in (ord("q"), 27):  # Q 或 ESC
            print("[INFO] 使用者結束收集。")
            break
        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"[INFO] 自動連拍：{'開啟' if auto_mode else '關閉'}")
        elif key == ord(" "):
            _save_frame(frame, save_dir, existing + saved)
            saved += 1
            print(f"[INFO] 已儲存 {saved}/{count} 張")

        # ── 自動連拍 ──────────────────────────────────────────────────────
        if auto_mode:
            now = time.time()
            if now - last_auto_time >= 0.5:
                _save_frame(frame, save_dir, existing + saved)
                saved += 1
                last_auto_time = now
                print(f"[INFO] 自動拍攝 {saved}/{count} 張")

    cap.release()
    cv2.destroyAllWindows()

    total = existing + saved
    print(f"\n[DONE] {name} 共 {total} 張照片（本次新增 {saved} 張）")
    if total < 25:
        print(f"[WARN] 照片數量不足 25 張，建議補拍至少 {25 - total} 張。")


def _save_frame(frame, save_dir: Path, idx: int) -> None:
    """儲存單一影格為 JPEG。"""
    filename = save_dir / f"{idx:04d}.jpg"
    cv2.imwrite(str(filename), frame)


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Enrollment dataset 收集工具")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="授權人員姓名，例如 --name henry",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="目標收集張數（預設 30）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/enrollment",
        help="enrollment 根目錄（預設 data/enrollment）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    collect_faces(name=args.name, count=args.count, output_dir=output_dir)


if __name__ == "__main__":
    main()
