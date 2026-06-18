#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University — I4210 AI實務專題
# tests/test_recognizer.py — Unit tests for face recognition
"""Unit tests for src/recognition/recognizer.py."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 移除 conftest 對 src.recognition/detection 的 stub 改用真實模組，並 stub
# onnxruntime 讓真實 recognizer 能在無 onnxruntime 的 CI runner 上 import。
for _key in [k for k in sys.modules if "src.recognition" in k or "src.detection" in k]:
    sys.modules.pop(_key, None)
sys.modules.setdefault("onnxruntime", MagicMock())

from src.recognition.recognizer import FaceRecognizer  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_db(tmp_path):
    """Create a temporary face database for testing."""
    db = {
        "names": ["henry", "alice"],
        "embeddings": np.array(
            [
                [1.0] + [0.0] * 127,  # henry (1st-dim focus)
                [0.0, 1.0] + [0.0] * 126,  # alice (2nd-dim focus)
            ],
            dtype=np.float32,
        ),
    }
    db_path = tmp_path / "face_db.npy"
    np.save(str(db_path), db)
    return str(db_path)


@pytest.fixture
def mock_session():
    """Create a mock ONNX runtime InferenceSession."""
    session = MagicMock()
    # 預設回傳符合 henry 特徵的未歸一化向量
    embedding = np.array([[10.0] + [0.0] * 127], dtype=np.float32)
    session.run.return_value = [embedding]
    return session


@pytest.fixture
def recognizer(mock_session, mock_db):
    """Yield a managed FaceRecognizer instance with patched session."""
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        r = FaceRecognizer(onnx_path="fake.onnx", db_path=mock_db, threshold=0.85)
        yield r


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestFaceRecognizer:
    def test_load_db(self, recognizer):
        """Verify that the face database loads properly."""
        assert recognizer.names == ["henry", "alice"]
        assert recognizer.embeddings.shape == (2, 128)

    def test_get_embedding_normalized(self, recognizer):
        """Verify that extracted embedding is properly L2 normalized."""
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        emb = recognizer.get_embedding(dummy)
        assert emb.shape == (128,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_match_authorized(self, recognizer):
        """Verify successful match when similarity exceeds threshold."""
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        result = recognizer.match(dummy)
        assert result.name == "henry"
        assert result.authorized is True
        assert result.similarity > 0.85

    def test_match_unknown(self, mock_db):
        """Verify negative match when similarity is below threshold."""
        session = MagicMock()
        # 修正：回傳第 3 維度為主的特徵向量，使其與 henry(1)、alice(2) 完全正交
        # 歸一化後為 [0, 0, 1, 0...]，與資料庫兩人的相似度計算出來絕對是 0.0
        embedding = np.array([[0.0, 0.0, 10.0] + [0.0] * 125], dtype=np.float32)
        session.run.return_value = [embedding]

        with patch("onnxruntime.InferenceSession", return_value=session):
            r = FaceRecognizer(onnx_path="fake.onnx", db_path=mock_db, threshold=0.85)
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            result = r.match(dummy)

        # 相似度為 0.0 < 0.85，必定判定為陌生人
        assert result.authorized is False
        assert result.name == "unknown"

    def test_reload_db(self, recognizer, tmp_path):
        """Verify that reload_db dynamically updates authorized list."""
        new_db = {
            "names": ["bob"],
            "embeddings": np.array([[0.0] * 128], dtype=np.float32),
        }
        new_path = tmp_path / "new_db.npy"
        np.save(str(new_path), new_db)
        recognizer.reload_db(str(new_path))
        assert recognizer.names == ["bob"]
