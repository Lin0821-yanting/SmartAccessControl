#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""Unit tests for pipeline decision logic（TemporalVoter + make_decision）."""

from unittest.mock import MagicMock

import pytest


# ── Mock dataclasses ──────────────────────────────────────────────────────────
def make_recog(name="henry", similarity=0.95, authorized=True):
    r = MagicMock()
    r.name = name
    r.similarity = similarity
    r.authorized = authorized
    return r


def make_liveness(is_live=True, score=0.9):
    mock_live = MagicMock()
    mock_live.is_live = is_live
    mock_live.score = score
    return mock_live


# ── Import pipeline logic ─────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def import_pipeline(monkeypatch):
    """Mock 掉 pipeline 的所有硬體依賴，只測試邏輯。."""
    import sys
    from unittest.mock import MagicMock

    # Mock 所有會在 import 時觸發硬體的模組
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
        from src.pipeline.main import TemporalVoter

        self.voter = TemporalVoter(required_frames=2)

    def test_not_authorized_before_required_frames(self):
        """未達到連續幀數時不授權。."""
        authorized, _name = self.voter.vote("henry")
        assert authorized is False

    def test_authorized_after_required_frames(self):
        """連續 N 幀同一人 → 授權。."""
        self.voter.vote("henry")
        authorized, name = self.voter.vote("henry")
        assert authorized is True
        assert name == "henry"

    def test_unknown_never_authorized(self):
        """Unknown 永遠不授權。."""
        self.voter.vote("unknown")
        authorized, _name = self.voter.vote("unknown")
        assert authorized is False

    def test_reset_clears_buffer(self):
        """Reset 後重新計數。."""
        self.voter.vote("henry")
        self.voter.reset()
        authorized, _ = self.voter.vote("henry")
        assert authorized is False

    def test_different_names_not_authorized(self):
        """不同人交替出現不授權。."""
        self.voter.vote("henry")
        authorized, _ = self.voter.vote("alice")
        assert authorized is False


class TestMakeDecision:
    def setup_method(self):
        from src.pipeline.main import TemporalVoter, make_decision

        self.voter = TemporalVoter(required_frames=2)
        self.make_decision = make_decision

    def test_spoof_rejected(self):
        """活體偵測失敗 → 拒絕。."""
        recog = make_recog(authorized=True)
        liveness = make_liveness(is_live=False, score=0.3)
        granted, _name, reason = self.make_decision(recog, liveness, self.voter)
        assert granted is False
        assert "spoof" in reason

    def test_low_similarity_rejected(self):
        """相似度不足 → 拒絕。."""
        recog = make_recog(authorized=False, similarity=0.5)
        liveness = make_liveness(is_live=True)
        granted, _name, reason = self.make_decision(recog, liveness, self.voter)
        assert granted is False
        assert "similarity" in reason

    def test_granted_after_consecutive_frames(self):
        """連續 2 幀通過 → 授權。."""
        recog = make_recog(name="henry", authorized=True)
        liveness = make_liveness(is_live=True)
        self.make_decision(recog, liveness, self.voter)
        granted, name, _reason = self.make_decision(recog, liveness, self.voter)
        assert granted is True
        assert name == "henry"

    def test_voter_reset_on_spoof(self):
        """Spoof 偵測後 voter 重置。."""
        recog = make_recog(authorized=True)
        liveness = make_liveness(is_live=True)
        self.make_decision(recog, liveness, self.voter)  # 第 1 幀

        # spoof 攻擊
        self.make_decision(recog, make_liveness(is_live=False), self.voter)

        # 重置後需要重新累積
        granted, _, _ = self.make_decision(recog, liveness, self.voter)
        assert granted is False
