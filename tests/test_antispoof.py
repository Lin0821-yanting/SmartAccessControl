import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# 1. 物理洗白快取
for key in list(sys.modules.keys()):
    if "src.antispoof" in key:
        sys.modules.pop(key, None)

# 2. 正常導入
from src.antispoof.antispoof import AntiSpoof


@pytest.fixture
def mock_onnx_session_real():
    """建立一個回傳真人分數的 ONNX InferenceSession Mock。"""
    session = MagicMock()
    # 根據實作：第二個數值大代表真人 (real)。對調成 [-10.0, 10.0]
    session.run.return_value = [np.array([[-10.0, 10.0]], dtype=np.float32)]
    return session


@pytest.fixture
def mock_onnx_session_spoof():
    """建立一個回傳欺騙翻拍分數的 ONNX InferenceSession Mock。"""
    session = MagicMock()
    # 根據實作：第一個數值大代表偽造 (spoof)。對調成 [10.0, -10.0]
    session.run.return_value = [np.array([[10.0, -10.0]], dtype=np.float32)]
    return session


class TestAntiSpoof:
    def test_predict_real(self, mock_onnx_session_real):
        """測試真實人臉通過。"""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session_real):
            detector = AntiSpoof(onnx_path="fake.onnx", threshold=0.6)
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            result = detector.predict(dummy_img)

        assert result.is_live is True

    def test_predict_spoof(self, mock_onnx_session_spoof):
        """測試照片/螢幕欺騙攻擊。"""
        with patch(
            "onnxruntime.InferenceSession", return_value=mock_onnx_session_spoof
        ):
            detector = AntiSpoof(onnx_path="fake.onnx", threshold=0.6)
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            result = detector.predict(dummy_img)

        assert result.is_live is False

    def test_score_range(self, mock_onnx_session_real):
        """驗證分數是否落在 0.0 ~ 1.0。"""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session_real):
            detector = AntiSpoof(onnx_path="fake.onnx", threshold=0.6)
            dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
            result = detector.predict(dummy_img)

        assert 0.0 <= result.score <= 1.0

    def test_input_resize(self, mock_onnx_session_real):
        """任意大小的輸入都能正常處理。"""
        with patch("onnxruntime.InferenceSession", return_value=mock_onnx_session_real):
            detector = AntiSpoof(onnx_path="fake.onnx", threshold=0.6)
            large = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.predict(large)

        assert result is not None
