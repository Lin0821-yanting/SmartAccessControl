#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University — I4210 AI實務專題
# tests/test_pipeline.py — Unit tests for pipeline decision logic
"""Unit tests for pipeline decision logic (TemporalVoter + make_decision)."""

import sys

# 1. 強制洗清全域 Mock 污染緩存，確保拿到真實的 main 邏輯
for key in list(sys.modules.keys()):
    if "src.pipeline" in key:
        sys.modules.pop(key, None)

from unittest.mock import MagicMock
import numpy as np
import pytest


# ── Mock dataclasses ──────────────────────────────────────────────────────────
def make_recog(name="henry", similarity=0.95, authorized=True):
    """Create a mock recognition result."""
    r = MagicMock()
    r.name = name
    r.similarity = similarity
    r.authorized = authorized
    return r


def make_liveness(is_live=True, score=0.9):
    """Create a mock liveness result."""
    mock_live = MagicMock()
    mock_live.is_live = is_live
    mock_live.score = score
    return mock_live


# ── Import pipeline logic ─────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def import_pipeline(monkeypatch):
    """Mock out heavy third-party hardware and AI dependencies."""
    # Mock 所有會在 import 時觸發硬體或重量級的模組
    for mod in [
        "cv2",
        "ultralytics",
        "tensorrt",
        "torch",
        "onnxruntime",
        "pycuda",
        "pycuda.autoinit",
        "mediapipe",
        "Jetson",
        "Jetson.GPIO",
    ]:
        sys.modules.setdefault(mod, MagicMock())


class TestTemporalVoter:
    def setup_method(self):
        """Set up TemporalVoter instance before each test."""
        from src.pipeline.main import TemporalVoter

        self.voter = TemporalVoter(required_frames=2)

    def test_not_authorized_before_required_frames(self):
        """Verify access is denied before reaching required frame count."""
        authorized, _name = self.voter.vote("henry")
        assert authorized is False

    def test_authorized_after_required_frames(self):
        """Verify access is granted after N consecutive identical frames."""
        self.voter.vote("henry")
        authorized, name = self.voter.vote("henry")
        assert authorized is True
        assert name == "henry"

    def test_unknown_never_authorized(self):
        """Verify unknown identity is never authorized."""
        self.voter.vote("unknown")
        authorized, _name = self.voter.vote("unknown")
        assert authorized is False

    def test_reset_clears_buffer(self):
        """Verify reset clears the internal voting buffer."""
        self.voter.vote("henry")
        self.voter.reset()
        authorized, _ = self.voter.vote("henry")
        assert authorized is False

    def test_different_names_not_authorized(self):
        """Verify changing names disrupts consecutive frame logic."""
        self.voter.vote("henry")
        authorized, _ = self.voter.vote("alice")
        assert authorized is False


class TestMakeDecision:
    def setup_method(self):
        """Set up environment for decision engine tests."""
        from src.pipeline.main import TemporalVoter, make_decision

        self.voter = TemporalVoter(required_frames=2)
        self.make_decision = make_decision

    def test_spoof_rejected(self):
        """Verify spoofing attack results in access denial."""
        recog = make_recog(authorized=True)
        liveness = make_liveness(is_live=False, score=0.3)
        granted, _name, reason = self.make_decision(recog, liveness, self.voter)
        assert granted is False
        assert "spoof" in reason

    def test_low_similarity_rejected(self):
        """Verify low confidence match results in access denial."""
        recog = make_recog(authorized=False, similarity=0.5)
        liveness = make_liveness(is_live=True)
        granted, _name, reason = self.make_decision(recog, liveness, self.voter)
        assert granted is False
        assert "similarity" in reason

    def test_granted_after_consecutive_frames(self):
        """Verify successful grant after 2 consecutive passing frames."""
        recog = make_recog(name="henry", authorized=True)
        liveness = make_liveness(is_live=True)
        self.make_decision(recog, liveness, self.voter)
        granted, name, _reason = self.make_decision(recog, liveness, self.voter)
        assert granted is True
        assert name == "henry"

    def test_voter_reset_on_spoof(self):
        """Verify voter state resets immediately upon spoof detection."""
        recog = make_recog(authorized=True)
        liveness = make_liveness(is_live=True)
        self.make_decision(recog, liveness, self.voter)

        # Spoof attack triggers reset
        self.make_decision(recog, make_liveness(is_live=False), self.voter)

        # Must restart accumulation from zero
        granted, _, _ = self.make_decision(recog, liveness, self.voter)
        assert granted is False


# ── Integration Test to Maximize Coverage ─────────────────────────────────────
def test_pipeline_main_happy_path_decision():
    """Let data flow through pipeline/main.py decision logic to maximize coverage."""
    from src.pipeline.main import TemporalVoter, make_decision

    # 1. 建立一個滿足所有通過條件的情境
    voter = TemporalVoter(required_frames=1)

    recog = make_recog(name="henry", authorized=True, similarity=0.92)
    liveness = make_liveness(is_live=True, score=0.88)

    # 2. 第一次觸發決策，累積幀數
    granted, name, reason = make_decision(recog, liveness, voter)

    # 3. 修正斷言：精準比對實際回傳的資訊字串
    assert granted is True
    assert name == "henry"
    assert "similarity" in reason
    assert "liveness" in reason


def test_pipeline_main_rejected_cases_coverage():
    """觸發 make_decision 內部其他未被覆蓋的錯誤處理分支。"""
    from src.pipeline.main import TemporalVoter, make_decision

    # 測試 1：陌生人拒絕路徑
    voter_unknown = TemporalVoter(required_frames=1)
    recog_unknown = make_recog(name="unknown", authorized=False, similarity=0.3)
    liveness_ok = make_liveness(is_live=True, score=0.9)
    granted, name, reason = make_decision(recog_unknown, liveness_ok, voter_unknown)
    assert granted is False

    # 測試 2：活體偵測失敗重置路徑
    voter_spoof = TemporalVoter(required_frames=1)
    recog_ok = make_recog(name="henry", authorized=True, similarity=0.95)
    liveness_spoof = make_liveness(is_live=False, score=0.1)
    granted, name, reason = make_decision(recog_ok, liveness_spoof, voter_spoof)
    assert granted is False
