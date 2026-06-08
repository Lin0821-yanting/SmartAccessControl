#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""Unit tests for src/antispoof/antispoof.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_session_real():
    """Mock session 回傳 real logits（real > spoof）。."""
    session = MagicMock()
    session.run.return_value = [np.array([[-10.0, 10.0]], dtype=np.float32)]
    return session


@pytest.fixture
def mock_session_spoof():
    """Mock session 回傳 spoof logits（spoof > real）。."""
    session = MagicMock()
    session.run.return_value = [np.array([[10.0, -10.0]], dtype=np.float32)]
    return session


@pytest.fixture
def antispoof_real(mock_session_real):
    with patch("onnxruntime.InferenceSession", return_value=mock_session_real):
        from src.antispoof.antispoof import AntiSpoof

        return AntiSpoof(onnx_path="fake.onnx", threshold=0.6)


@pytest.fixture
def antispoof_spoof(mock_session_spoof):
    with patch("onnxruntime.InferenceSession", return_value=mock_session_spoof):
        from src.antispoof.antispoof import AntiSpoof

        return AntiSpoof(onnx_path="fake.onnx", threshold=0.6)


class TestAntiSpoof:
    def test_predict_real(self, antispoof_real):
        """Real logits → is_live=True, label='real'。."""
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        result = antispoof_real.predict(dummy)
        assert result.is_live is True
        assert result.label == "real"
        assert result.score > 0.6

    def test_predict_spoof(self, antispoof_spoof):
        """Spoof logits → is_live=False, label='spoof'。."""
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        result = antispoof_spoof.predict(dummy)
        assert result.is_live is False
        assert result.label == "spoof"
        assert result.score < 0.6

    def test_score_range(self, antispoof_real):
        """Score 必須在 0~1 之間。."""
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        result = antispoof_real.predict(dummy)
        assert 0.0 <= result.score <= 1.0

    def test_input_resize(self, antispoof_real):
        """任意大小的輸入都能正常處理。."""
        large = np.zeros((480, 640, 3), dtype=np.uint8)
        result = antispoof_real.predict(large)
        assert result is not None
