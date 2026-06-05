#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
recognizer.py
-------------
MobileFaceNet ONNX Runtime CUDA 推論器 + Cosine Similarity 身分比對。
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class RecognitionResult:
    name: str
    similarity: float
    authorized: bool


class FaceRecognizer:
    def __init__(
        self,
        onnx_path: str = "models/weights/MobileFaceNet.onnx",
        db_path: str   = "data/face_db.npy",
        threshold: float = 0.85,
    ):
        self.threshold = threshold
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._load_db(db_path)
        print(f"[FaceRecognizer] 載入 ONNX：{onnx_path}")
        print(f"[FaceRecognizer] 授權人員：{self.names}")

    def _load_db(self, db_path: str) -> None:
        db = np.load(db_path, allow_pickle=True).item()
        self.names: list            = db["names"]
        self.embeddings: np.ndarray = db["embeddings"]

    def reload_db(self, db_path: str) -> None:
        self._load_db(db_path)
        print(f"[FaceRecognizer] face_db 重新載入：{self.names}")

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_crop, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[np.newaxis])
        emb = self.sess.run(["output0"], {"input0": img})[0][0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def match(self, face_crop: np.ndarray) -> RecognitionResult:
        emb  = self.get_embedding(face_crop)
        sims = self.embeddings @ emb
        idx  = int(np.argmax(sims))
        sim  = float(sims[idx])
        if sim >= self.threshold:
            return RecognitionResult(name=self.names[idx], similarity=sim, authorized=True)
        return RecognitionResult(name="unknown", similarity=sim, authorized=False)


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/enrollment/henry/0000.jpg"
    r   = FaceRecognizer()
    img = cv2.imread(img_path)
    res = r.match(img)
    print(f"辨識：{res.name}  similarity={res.similarity:.4f}  authorized={res.authorized}")