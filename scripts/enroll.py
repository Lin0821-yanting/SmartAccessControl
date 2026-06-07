#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
enroll.py
---------
讀取 data/enrollment/<name>/*.jpg，
用 YOLOv8n-face 偵測並裁切人臉，
再用 MobileFaceNet ONNX 產生 128-dim embedding，
L2 正規化後取平均，儲存至 data/face_db.npy。

Usage:
    pdm run python scripts/enroll.py
    pdm run python scripts/enroll.py --name henry
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def build_face_db(
    enrollment_dir: Path,
    db_path: Path,
    yolo_weights: str = "models/weights/yolov8n-face.pt",
    facenet_onnx: str = "models/weights/MobileFaceNet.onnx",
    min_photos: int = 10,
    target_name: str = None,
) -> None:
    print(f"\n[INFO] 載入 YOLOv8n-face：{yolo_weights}")
    detector = YOLO(yolo_weights, task="pose")

    print(f"[INFO] 載入 MobileFaceNet：{facenet_onnx}")
    sess = ort.InferenceSession(
        facenet_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # 載入現有 face_db（若存在）
    if db_path.exists():
        existing = np.load(str(db_path), allow_pickle=True).item()
        names = list(existing["names"])
        embeddings = list(existing["embeddings"])
        print(f"[INFO] 現有 face_db：{names}")
    else:
        names, embeddings = [], []

    # 掃描人員資料夾
    person_dirs = sorted([d for d in enrollment_dir.iterdir() if d.is_dir()])
    if target_name:
        person_dirs = [d for d in person_dirs if d.name == target_name]

    if not person_dirs:
        raise RuntimeError(f"找不到人員資料夾：{enrollment_dir}")

    for person_dir in person_dirs:
        images = sorted(list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")))
        if len(images) < min_photos:
            print(f"[WARN] {person_dir.name} 只有 {len(images)} 張，跳過。")
            continue

        print(f"\n[INFO] 處理：{person_dir.name}（{len(images)} 張）")
        person_embeddings = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 偵測人臉並裁切
            results = detector(img, imgsz=416, conf=0.5, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) == 0:
                print(f"  [WARN] 未偵測到人臉：{img_path.name}")
                continue

            x1, y1, x2, y2 = boxes[0].astype(int)
            crop = img[y1:y2, x1:x2]

            # MobileFaceNet 推論
            face = cv2.resize(crop, (112, 112))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            face = np.transpose(face, (2, 0, 1))[np.newaxis]
            emb = sess.run(["output0"], {"input0": face})[0][0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            person_embeddings.append(emb)

        if not person_embeddings:
            print(f"  [WARN] {person_dir.name} 無有效人臉，跳過。")
            continue

        avg_emb = np.mean(person_embeddings, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

        # 更新或新增
        if person_dir.name in names:
            idx = names.index(person_dir.name)
            embeddings[idx] = avg_emb
            print(f"  [UPDATE] {person_dir.name} embedding 已更新")
        else:
            names.append(person_dir.name)
            embeddings.append(avg_emb)
            print(f"  [ADD] {person_dir.name} 新增至 face_db")

        print(f"  embedding norm：{np.linalg.norm(avg_emb):.4f}，使用 {len(person_embeddings)} 張")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(db_path), {"names": names, "embeddings": np.array(embeddings)})
    print(f"\n[DONE] face_db 儲存：{db_path}，共 {len(names)} 位：{names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrollment：建立人臉特徵資料庫")
    parser.add_argument("--enrollment_dir", type=str, default="data/enrollment")
    parser.add_argument("--db_path", type=str, default="data/face_db.npy")
    parser.add_argument("--yolo", type=str, default="models/weights/yolov8n-face.pt")
    parser.add_argument("--facenet", type=str, default="models/weights/MobileFaceNet.onnx")
    parser.add_argument("--min_photos", type=int, default=10)
    parser.add_argument("--name", type=str, default=None, help="只處理指定人員")
    args = parser.parse_args()

    build_face_db(
        enrollment_dir=Path(args.enrollment_dir),
        db_path=Path(args.db_path),
        yolo_weights=args.yolo,
        facenet_onnx=args.facenet,
        min_photos=args.min_photos,
        target_name=args.name,
    )


if __name__ == "__main__":
    main()
