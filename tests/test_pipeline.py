#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University — I4210 AI實務專題
# tests/test_pipeline.py — Unit tests for src/pipeline/main.py decision logic
"""Unit tests for the pipeline decision logic (TemporalVoter + make_decision).

Heavy AI / GPIO modules (cv2, ultralytics, Jetson.GPIO, src.detection, ...) are
stubbed in tests/conftest.py, so importing src.pipeline.main pulls in the real
TemporalVoter and make_decision without touching any hardware.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.pipeline.main import (
    _LIVENESS_THRESHOLD,
    _SPOOF_CONFIRM_FRAMES,
    TemporalVoter,
    make_decision,
)

# Liveness scores clearly above / below the SPOOF threshold.
_LIVE = _LIVENESS_THRESHOLD + 0.1
_FAIL = _LIVENESS_THRESHOLD - 0.1


def _recog(name="henry", similarity=0.95, authorized=True):
    """Build a mock RecognitionResult."""
    r = MagicMock()
    r.name = name
    r.similarity = similarity
    r.authorized = authorized
    return r


def _liveness(score=_LIVE):
    """Build a mock LivenessResult (make_decision compares .score to threshold)."""
    m = MagicMock()
    m.score = score
    return m


class TestTemporalVoter:
    """連續幀投票邏輯."""

    def test_not_authorized_before_required_frames(self):
        voter = TemporalVoter(required_frames=2)
        authorized, _name = voter.vote("henry")
        assert authorized is False

    def test_authorized_after_required_frames(self):
        voter = TemporalVoter(required_frames=2)
        voter.vote("henry")
        authorized, name = voter.vote("henry")
        assert authorized is True
        assert name == "henry"

    def test_unknown_never_authorized(self):
        voter = TemporalVoter(required_frames=2)
        voter.vote("unknown")
        authorized, _name = voter.vote("unknown")
        assert authorized is False

    def test_reset_clears_buffer(self):
        voter = TemporalVoter(required_frames=2)
        voter.vote("henry")
        voter.reset()
        authorized, _name = voter.vote("henry")
        assert authorized is False

    def test_different_names_not_authorized(self):
        voter = TemporalVoter(required_frames=2)
        voter.vote("henry")
        authorized, _name = voter.vote("alice")
        assert authorized is False


class TestMakeDecision:
    """make_decision 的五種判決與狀態轉移（對應 run_pipeline Stage 4）."""

    def test_grant_after_required_frames(self):
        voter = TemporalVoter(required_frames=2)
        recog, live = _recog(authorized=True), _liveness(_LIVE)
        d1, streak = make_decision(recog, live, voter, 0)
        assert d1 == "DENY"  # first qualifying frame, not enough yet
        d2, _streak = make_decision(recog, live, voter, streak)
        assert d2 == "GRANT"

    def test_unknown_when_not_in_db(self):
        voter = TemporalVoter(required_frames=2)
        decision, _streak = make_decision(_recog(authorized=False), _liveness(_LIVE), voter, 0)
        assert decision == "UNKNOWN"

    def test_liveness_fail_below_streak_is_ignore(self):
        voter = TemporalVoter(required_frames=2)
        decision, streak = make_decision(_recog(authorized=True), _liveness(_FAIL), voter, 0)
        assert decision == "IGNORE"
        assert streak == 1

    def test_spoof_after_confirm_frames(self):
        voter = TemporalVoter(required_frames=2)
        decision, _streak = make_decision(
            _recog(authorized=True), _liveness(_FAIL), voter, _SPOOF_CONFIRM_FRAMES - 1
        )
        assert decision == "SPOOF"

    def test_liveness_fail_resets_voter(self):
        voter = TemporalVoter(required_frames=3)
        recog = _recog(authorized=True)
        make_decision(recog, _liveness(_LIVE), voter, 0)
        assert voter.consecutive_frames == 1
        make_decision(recog, _liveness(_FAIL), voter, 0)
        assert voter.consecutive_frames == 0

    def test_spoof_streak_resets_on_live_frame(self):
        voter = TemporalVoter(required_frames=2)
        recog = _recog(authorized=True)
        _d, streak = make_decision(recog, _liveness(_FAIL), voter, 3)
        assert streak == 4
        _d2, streak = make_decision(recog, _liveness(_LIVE), voter, streak)
        assert streak == 0
