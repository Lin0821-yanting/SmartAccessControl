#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
antispoof.py
------------
MiniFASNet ONNX Runtime CUDA 活體偵測推論器。
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class LivenessResult:
    is_live: bool
    score: float
    label: str


class AntiSpoof:
    def __init__(
        self,
        onnx_path: str   = "models/weights/minifasnet.onnx",
        threshold: float = 0.9,
        input_size: int  = 128,
    ):
        self.threshold  = threshold
        self.input_size = input_size
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"[AntiSpoof] 載入 ONNX：{onnx_path}")

    def predict(self, face_crop: np.ndarray) -> LivenessResult:
        img = cv2.resize(face_crop, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[np.newaxis])

        logits = self.sess.run(["output"], {"input": img})[0][0]
        logits = logits - np.max(logits)
        exp    = np.exp(logits)
        probs  = exp / exp.sum()

        real_prob = float(probs[1])
        is_live   = real_prob >= self.threshold
        return LivenessResult(is_live=is_live, score=real_prob, label="real" if is_live else "spoof")


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/enrollment/henry/0000.jpg"
    a   = AntiSpoof()
    img = cv2.imread(img_path)
    res = a.predict(img)
    print(f"活體：{res.label}  score={res.score:.4f}  is_live={res.is_live}")