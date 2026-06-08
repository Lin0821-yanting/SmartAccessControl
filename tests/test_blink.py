#!/usr/bin/env python3
# copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""Unit tests for src/antispoof/blink.py, Mock 掉 MediaPipe，純邏輯測試."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def make_landmark(x, y, z=0.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def make_face_landmarks(ear_value: float, w: int = 640, h: int = 480):
    """Create mock MediaPipe landmarks to match target EAR.

    Left eye indices: [33, 160, 158, 133, 153, 144].
    Formula: EAR = (A + B) / (2c).
    """
    lms = [MagicMock() for _ in range(468)]

    cx, cy = 0.5, 0.5
    c = 0.1  # 水平距離（歸一化）
    ab = ear_value * c  # 垂直距離

    # 左眼：33=左端, 133=右端, 160/144=上下, 158/153=上下
    lms[33] = make_landmark(cx - c / 2, cy)
    lms[133] = make_landmark(cx + c / 2, cy)
    lms[160] = make_landmark(cx, cy - ab)
    lms[144] = make_landmark(cx, cy + ab)
    lms[158] = make_landmark(cx, cy - ab)
    lms[153] = make_landmark(cx, cy + ab)

    # 右眼：362=左端, 263=右端
    lms[362] = make_landmark(cx - c / 2, cy)
    lms[263] = make_landmark(cx + c / 2, cy)
    lms[385] = make_landmark(cx, cy - ab)
    lms[380] = make_landmark(cx, cy + ab)
    lms[387] = make_landmark(cx, cy - ab)
    lms[373] = make_landmark(cx, cy + ab)

    return lms


@pytest.fixture
def mock_mp():
    """Mock MediaPipe Face Mesh。."""
    with patch("mediapipe.solutions.face_mesh") as mock_fm:
        mock_instance = MagicMock()
        mock_fm.FaceMesh.return_value.__enter__ = lambda s: mock_instance
        mock_fm.FaceMesh.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def detector():
    with patch("mediapipe.solutions.face_mesh"):
        from src.antispoof.blink import BlinkDetector

        d = BlinkDetector(ear_threshold=0.25, required_blinks=2, consec_frames=2)
        return d


class TestBlinkDetector:
    def test_no_face_returns_default(self, detector, mock_mp):
        """未偵測到人臉時回傳預設值。."""
        detector.face_mesh = mock_mp
        mock_mp.process.return_value.multi_face_landmarks = None

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.update(dummy)
        assert result.ear == 1.0
        assert result.is_closed is False
        assert result.blink_count == 0

    def test_open_eye_not_closed(self, detector, mock_mp):
        """EAR > threshold → is_closed=False。."""
        detector.face_mesh = mock_mp
        lms = make_face_landmarks(ear_value=0.35)
        face = MagicMock()
        face.landmark = lms
        mock_mp.process.return_value.multi_face_landmarks = [face]

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.update(dummy)
        assert result.is_closed is False

    def test_reset_clears_count(self, detector):
        """Reset 後眨眼計數歸零。."""
        detector._blink_count = 3
        detector.reset()
        assert detector._blink_count == 0

    def test_blink_confirmed_after_required(self, detector):
        """達到 required_blinks 後 blink_confirmed=True。."""
        detector._blink_count = 2
        result = MagicMock()
        result.blink_count = 2
        result.blink_confirmed = detector._blink_count >= detector.required_blinks
        assert result.blink_confirmed is True

    def test_timeout_resets_count(self, detector):
        """超過 reset_timeout 後計數重置。."""
        detector._blink_count = 1
        detector._last_blink_t = time.time() - 15  # 模擬 15 秒前

        with patch("mediapipe.solutions.face_mesh"):
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            detector.face_mesh = MagicMock()
            detector.face_mesh.process.return_value.multi_face_landmarks = None
            detector.update(dummy)

        assert detector._blink_count == 0
