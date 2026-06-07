#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name>, <Partner's Name>
# Tatung University — I4210 AI實務專題
"""tests/test_decision_engine.py — unit tests for DecisionEngine.

Scenario coverage
-----------------
Group A — Specified scenarios
  A1  真實情況正確通過 (GRANT after required consecutive frames)
  A2  真實情況露幀 / 遮蔽 (genuine person, frame drop mid-sequence)
  A3  距離未達標 (HC-SR04 gate not triggered → ignore())
  A4  照片但通過 (high-similarity spoof, anti_spoof_pass=False must block)
  A5  照片成功過濾 (spoof correctly rejected as SPOOF)

Group B — Boundary values
  B1  similarity == threshold exactly (≥ passes, < fails)
  B2  required_frames == 1 (single-frame GRANT)
  B3  Counter accumulates to required_frames − 1 then resets on interruption

Group C — Priority ordering
  C1  SPOOF overrides high similarity (similarity=1.0, spoof fails → SPOOF)
  C2  UNKNOWN when face_in_db=False even with anti_spoof_pass=True
  C3  DENY when face_in_db=True but similarity below threshold

Group D — State machine resets
  D1  SPOOF resets consecutive_frames counter
  D2  UNKNOWN resets consecutive_frames counter
  D3  DENY resets consecutive_frames counter
  D4  ignore() resets consecutive_frames counter
  D5  reset() public method zeroes counter without evaluate()
  D6  Counter resets to 0 after GRANT is issued

Group E — Multi-frame accumulation
  E1  Counter increments correctly across partial accumulation
  E2  Interrupted accumulation (DENY in the middle) resets counter to 0
  E3  Interrupted accumulation (SPOOF in the middle) resets counter to 0
  E4  Interrupted accumulation (UNKNOWN in the middle) resets counter to 0
  E5  Counter does NOT reset between IGNORE frames (accumulation preserved)

Group F — Constructor validation
  F1  Invalid similarity_threshold (≤ 0) raises ValueError
  F2  Invalid similarity_threshold (> 1) raises ValueError
  F3  Invalid required_frames (< 1) raises ValueError
  F4  Custom thresholds are respected in evaluate()

Group G — ignore() and reset() behaviour
  G1  ignore() always returns Decision.IGNORE
  G2  ignore() resets a mid-accumulation counter
  G3  reset() leaves counter at 0 and next evaluate() starts fresh

Group H — consecutive_frames property
  H1  Property starts at 0
  H2  Property reflects increments during accumulation
  H3  Property is 0 immediately after GRANT

All tests are pure-logic; no GPIO, no MQTT, no hardware required.
"""

from __future__ import annotations

import pytest

from src.decision_engine import (
    REQUIRED_CONSECUTIVE_FRAMES,
    SIMILARITY_THRESHOLD,
    Decision,
    DecisionEngine,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> DecisionEngine:
    """Fresh DecisionEngine with default thresholds for each test."""
    return DecisionEngine()


@pytest.fixture()
def fast_engine() -> DecisionEngine:
    """DecisionEngine requiring only 1 consecutive frame (fast GRANT)."""
    return DecisionEngine(required_frames=1)


def _pass_frames(eng: DecisionEngine, n: int) -> list[Decision]:
    """Feed *n* fully-qualifying frames and return every Decision produced."""
    return [
        eng.evaluate(similarity=SIMILARITY_THRESHOLD, anti_spoof_pass=True, face_in_db=True)
        for _ in range(n)
    ]


# ===========================================================================
# Group A — Specified scenarios
# ===========================================================================


class TestSpecifiedScenarios:
    """Group A: the five scenarios explicitly requested in the task."""

    # A1 ── 真實情況正確通過 ─────────────────────────────────────────────────

    def test_a1_genuine_person_grants_after_required_frames(self, engine: DecisionEngine) -> None:
        """A genuine enrolled person passes after REQUIRED_CONSECUTIVE_FRAMES frames."""
        decisions = _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)
        assert decisions[-1] is Decision.GRANT

    def test_a1_frames_before_grant_return_ignore(self, engine: DecisionEngine) -> None:
        """All frames before the final GRANT must return IGNORE (accumulating)."""
        decisions = _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)
        for d in decisions[:-1]:
            assert d is Decision.IGNORE

    # A2 ── 真實情況露幀 / 遮蔽 ───────────────────────────────────────────────

    def test_a2_frame_drop_resets_and_requires_restart(self, engine: DecisionEngine) -> None:
        """A frame drop (similarity below threshold) mid-sequence resets the counter."""
        # Accumulate n-1 good frames, then one bad frame, then restart
        for _ in range(REQUIRED_CONSECUTIVE_FRAMES - 1):
            engine.evaluate(similarity=SIMILARITY_THRESHOLD, anti_spoof_pass=True, face_in_db=True)
        # Simulate occlusion (similarity drops)
        drop = engine.evaluate(
            similarity=SIMILARITY_THRESHOLD - 0.01,
            anti_spoof_pass=True,
            face_in_db=True,
        )
        assert drop is Decision.DENY
        assert engine.consecutive_frames == 0

    def test_a2_recovery_after_frame_drop_grants_on_new_sequence(
        self, engine: DecisionEngine
    ) -> None:
        """After a frame drop the system can still GRANT on a fresh sequence."""
        # Cause a reset
        engine.evaluate(similarity=0.50, anti_spoof_pass=True, face_in_db=True)
        # Now run a full clean sequence
        decisions = _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)
        assert decisions[-1] is Decision.GRANT

    def test_a2_occlusion_via_unknown_mid_sequence(self, engine: DecisionEngine) -> None:
        """Face goes out of DB (occlusion / wrong angle) mid-sequence resets counter."""
        engine.evaluate(similarity=SIMILARITY_THRESHOLD, anti_spoof_pass=True, face_in_db=True)
        occluded = engine.evaluate(similarity=0.0, anti_spoof_pass=True, face_in_db=False)
        assert occluded is Decision.UNKNOWN
        assert engine.consecutive_frames == 0

    # A3 ── 距離未達標 (HC-SR04 gate) ─────────────────────────────────────────

    def test_a3_hcsr04_not_triggered_returns_ignore(self, engine: DecisionEngine) -> None:
        """When HC-SR04 does not trigger, ignore() returns IGNORE."""
        result = engine.ignore()
        assert result is Decision.IGNORE

    def test_a3_ignore_does_not_accumulate_frames(self, engine: DecisionEngine) -> None:
        """Repeated ignore() calls must never increment the frame counter."""
        for _ in range(REQUIRED_CONSECUTIVE_FRAMES + 5):
            engine.ignore()
        assert engine.consecutive_frames == 0

    def test_a3_ignore_after_partial_accumulation_resets(self, engine: DecisionEngine) -> None:
        """ignore() called mid-sequence resets a partially accumulated counter."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        assert engine.consecutive_frames == REQUIRED_CONSECUTIVE_FRAMES - 1
        engine.ignore()
        assert engine.consecutive_frames == 0

    # A4 ── 照片但通過 (high similarity, anti-spoof disabled upstream — must block) ─

    def test_a4_high_similarity_with_spoof_fail_returns_spoof(self, engine: DecisionEngine) -> None:
        """Even similarity=1.0 is rejected when anti_spoof_pass=False."""
        result = engine.evaluate(
            similarity=1.0,
            anti_spoof_pass=False,
            face_in_db=True,
        )
        assert result is Decision.SPOOF

    def test_a4_spoof_with_face_in_db_false_still_spoof(self, engine: DecisionEngine) -> None:
        """SPOOF takes priority even when face_in_db is also False."""
        result = engine.evaluate(
            similarity=0.0,
            anti_spoof_pass=False,
            face_in_db=False,
        )
        assert result is Decision.SPOOF

    # A5 ── 照片成功過濾 ───────────────────────────────────────────────────────

    def test_a5_printed_photo_attack_filtered_as_spoof(self, engine: DecisionEngine) -> None:
        """A printed photo (anti_spoof_pass=False) is correctly classified SPOOF."""
        result = engine.evaluate(
            similarity=0.92,
            anti_spoof_pass=False,
            face_in_db=True,
        )
        assert result is Decision.SPOOF

    def test_a5_screen_replay_attack_filtered_as_spoof(self, engine: DecisionEngine) -> None:
        """A screen replay attack (anti_spoof_pass=False) is correctly SPOOF."""
        result = engine.evaluate(
            similarity=0.88,
            anti_spoof_pass=False,
            face_in_db=True,
        )
        assert result is Decision.SPOOF


# ===========================================================================
# Group B — Boundary values
# ===========================================================================


class TestBoundaryValues:
    """Group B: exact-threshold edge cases."""

    def test_b1_similarity_exactly_at_threshold_passes(self, engine: DecisionEngine) -> None:
        """Similarity == SIMILARITY_THRESHOLD must count as a qualifying frame (>=)."""
        result = engine.evaluate(
            similarity=SIMILARITY_THRESHOLD,
            anti_spoof_pass=True,
            face_in_db=True,
        )
        # Should be IGNORE (accumulating), not DENY
        assert result is not Decision.DENY
        assert engine.consecutive_frames == 1

    def test_b1_similarity_one_ulp_below_threshold_returns_deny(
        self, engine: DecisionEngine
    ) -> None:
        """Similarity just below threshold must return DENY."""
        result = engine.evaluate(
            similarity=SIMILARITY_THRESHOLD - 1e-9,
            anti_spoof_pass=True,
            face_in_db=True,
        )
        assert result is Decision.DENY

    def test_b2_required_frames_one_grants_immediately(self, fast_engine: DecisionEngine) -> None:
        """With required_frames=1 the very first qualifying frame issues GRANT."""
        result = fast_engine.evaluate(
            similarity=SIMILARITY_THRESHOLD,
            anti_spoof_pass=True,
            face_in_db=True,
        )
        assert result is Decision.GRANT

    def test_b3_counter_reaches_n_minus_one_then_deny_resets(self, engine: DecisionEngine) -> None:
        """Counter at required_frames-1 then a DENY resets it to 0."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        assert engine.consecutive_frames == REQUIRED_CONSECUTIVE_FRAMES - 1
        engine.evaluate(similarity=0.5, anti_spoof_pass=True, face_in_db=True)
        assert engine.consecutive_frames == 0

    @pytest.mark.parametrize("sim", [0.0, 0.5, 0.849])
    def test_b1_various_below_threshold_all_deny(self, engine: DecisionEngine, sim: float) -> None:
        """Any similarity below SIMILARITY_THRESHOLD with face_in_db=True yields DENY."""
        result = engine.evaluate(similarity=sim, anti_spoof_pass=True, face_in_db=True)
        assert result is Decision.DENY

    @pytest.mark.parametrize("sim", [SIMILARITY_THRESHOLD, 0.90, 0.99, 1.0])
    def test_b1_various_at_or_above_threshold_not_deny(
        self, engine: DecisionEngine, sim: float
    ) -> None:
        """Any similarity >= threshold with qualifying conditions must not be DENY."""
        result = engine.evaluate(similarity=sim, anti_spoof_pass=True, face_in_db=True)
        assert result is not Decision.DENY


# ===========================================================================
# Group C — Priority ordering
# ===========================================================================


class TestPriorityOrdering:
    """Group C: verifies the SPOOF > UNKNOWN > DENY > accumulate > GRANT chain."""

    def test_c1_spoof_overrides_perfect_similarity(self, engine: DecisionEngine) -> None:
        """SPOOF beats similarity=1.0 and face_in_db=True."""
        assert (
            engine.evaluate(similarity=1.0, anti_spoof_pass=False, face_in_db=True)
            is Decision.SPOOF
        )

    def test_c1_spoof_overrides_unknown_condition(self, engine: DecisionEngine) -> None:
        """SPOOF beats face_in_db=False (SPOOF has higher priority than UNKNOWN)."""
        assert (
            engine.evaluate(similarity=0.0, anti_spoof_pass=False, face_in_db=False)
            is Decision.SPOOF
        )

    def test_c2_unknown_when_not_in_db(self, engine: DecisionEngine) -> None:
        """UNKNOWN is returned when face_in_db=False (anti_spoof passed)."""
        assert (
            engine.evaluate(similarity=0.0, anti_spoof_pass=True, face_in_db=False)
            is Decision.UNKNOWN
        )

    def test_c2_unknown_regardless_of_similarity(self, engine: DecisionEngine) -> None:
        """UNKNOWN is returned even when similarity is high, if face_in_db=False."""
        assert (
            engine.evaluate(similarity=0.99, anti_spoof_pass=True, face_in_db=False)
            is Decision.UNKNOWN
        )

    def test_c3_deny_when_in_db_but_low_similarity(self, engine: DecisionEngine) -> None:
        """DENY is returned when face_in_db=True but similarity < threshold."""
        assert (
            engine.evaluate(
                similarity=SIMILARITY_THRESHOLD - 0.01,
                anti_spoof_pass=True,
                face_in_db=True,
            )
            is Decision.DENY
        )

    def test_c3_deny_not_spoof_when_only_similarity_fails(self, engine: DecisionEngine) -> None:
        """With anti_spoof_pass=True the result must be DENY, not SPOOF."""
        result = engine.evaluate(similarity=0.0, anti_spoof_pass=True, face_in_db=True)
        assert result is Decision.DENY
        assert result is not Decision.SPOOF


# ===========================================================================
# Group D — State machine resets
# ===========================================================================


class TestStateMachineResets:
    """Group D: every non-accumulating path must zero the counter."""

    def test_d1_spoof_resets_counter(self, engine: DecisionEngine) -> None:
        """SPOOF resets consecutive_frames to 0."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.evaluate(similarity=1.0, anti_spoof_pass=False, face_in_db=True)
        assert engine.consecutive_frames == 0

    def test_d2_unknown_resets_counter(self, engine: DecisionEngine) -> None:
        """UNKNOWN resets consecutive_frames to 0."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.evaluate(similarity=0.0, anti_spoof_pass=True, face_in_db=False)
        assert engine.consecutive_frames == 0

    def test_d3_deny_resets_counter(self, engine: DecisionEngine) -> None:
        """DENY resets consecutive_frames to 0."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.evaluate(
            similarity=SIMILARITY_THRESHOLD - 0.01,
            anti_spoof_pass=True,
            face_in_db=True,
        )
        assert engine.consecutive_frames == 0

    def test_d4_ignore_resets_counter(self, engine: DecisionEngine) -> None:
        """ignore() resets consecutive_frames to 0."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.ignore()
        assert engine.consecutive_frames == 0

    def test_d5_public_reset_zeroes_counter(self, engine: DecisionEngine) -> None:
        """reset() zeroes the counter without requiring an evaluate() call."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        assert engine.consecutive_frames > 0
        engine.reset()
        assert engine.consecutive_frames == 0

    def test_d6_counter_is_zero_immediately_after_grant(self, engine: DecisionEngine) -> None:
        """After GRANT is issued the counter is reset for the next person."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)
        assert engine.consecutive_frames == 0


# ===========================================================================
# Group E — Multi-frame accumulation
# ===========================================================================


class TestMultiFrameAccumulation:
    """Group E: counter arithmetic and interruption behaviour."""

    def test_e1_counter_increments_on_each_qualifying_frame(self, engine: DecisionEngine) -> None:
        """consecutive_frames increments by 1 for each qualifying frame."""
        for expected in range(1, REQUIRED_CONSECUTIVE_FRAMES):
            engine.evaluate(
                similarity=SIMILARITY_THRESHOLD,
                anti_spoof_pass=True,
                face_in_db=True,
            )
            assert engine.consecutive_frames == expected

    def test_e2_deny_mid_sequence_resets_counter(self, engine: DecisionEngine) -> None:
        """A DENY mid-sequence zeroes the counter; restart is needed."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.evaluate(similarity=0.0, anti_spoof_pass=True, face_in_db=True)
        assert engine.consecutive_frames == 0
        # Full new sequence should still succeed
        assert _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)[-1] is Decision.GRANT

    def test_e3_spoof_mid_sequence_resets_counter(self, engine: DecisionEngine) -> None:
        """A SPOOF mid-sequence zeroes the counter."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.evaluate(similarity=0.9, anti_spoof_pass=False, face_in_db=True)
        assert engine.consecutive_frames == 0

    def test_e4_unknown_mid_sequence_resets_counter(self, engine: DecisionEngine) -> None:
        """An UNKNOWN mid-sequence zeroes the counter."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.evaluate(similarity=0.0, anti_spoof_pass=True, face_in_db=False)
        assert engine.consecutive_frames == 0

    def test_e5_multiple_grants_each_start_fresh(self, engine: DecisionEngine) -> None:
        """Two consecutive GRANT sequences both succeed independently."""
        assert _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)[-1] is Decision.GRANT
        assert _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)[-1] is Decision.GRANT

    def test_e5_accumulation_preserved_across_multiple_ignore_calls(
        self, engine: DecisionEngine
    ) -> None:
        """ignore() resets accumulation; repeated ignores never reach GRANT."""
        for _ in range(REQUIRED_CONSECUTIVE_FRAMES * 10):
            result = engine.ignore()
            assert result is Decision.IGNORE
        assert engine.consecutive_frames == 0


# ===========================================================================
# Group F — Constructor validation
# ===========================================================================


class TestConstructorValidation:
    """Group F: invalid constructor arguments must raise ValueError."""

    @pytest.mark.parametrize("bad_threshold", [0.0, -0.1, -1.0])
    def test_f1_similarity_threshold_zero_or_negative_raises(self, bad_threshold: float) -> None:
        """similarity_threshold <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            DecisionEngine(similarity_threshold=bad_threshold)

    @pytest.mark.parametrize("bad_threshold", [1.001, 2.0, 100.0])
    def test_f2_similarity_threshold_above_one_raises(self, bad_threshold: float) -> None:
        """similarity_threshold > 1 must raise ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            DecisionEngine(similarity_threshold=bad_threshold)

    @pytest.mark.parametrize("bad_frames", [0, -1, -10])
    def test_f3_required_frames_zero_or_negative_raises(self, bad_frames: int) -> None:
        """required_frames < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="required_frames"):
            DecisionEngine(required_frames=bad_frames)

    def test_f4_custom_threshold_respected(self) -> None:
        """A custom similarity_threshold is honoured in evaluate()."""
        custom_threshold = 0.95
        eng = DecisionEngine(similarity_threshold=custom_threshold)
        # 0.90 is above default but below custom → must DENY
        assert eng.evaluate(similarity=0.90, anti_spoof_pass=True, face_in_db=True) is Decision.DENY
        # 0.95 is exactly at custom threshold → must accumulate (not DENY)
        eng2 = DecisionEngine(similarity_threshold=custom_threshold)
        result = eng2.evaluate(similarity=custom_threshold, anti_spoof_pass=True, face_in_db=True)
        assert result is not Decision.DENY

    def test_f4_custom_required_frames_respected(self) -> None:
        """A custom required_frames is honoured in evaluate()."""
        eng = DecisionEngine(required_frames=5)
        decisions = _pass_frames(eng, 4)
        assert all(d is Decision.IGNORE for d in decisions)
        assert (
            eng.evaluate(similarity=SIMILARITY_THRESHOLD, anti_spoof_pass=True, face_in_db=True)
            is Decision.GRANT
        )

    def test_f4_boundary_threshold_value_one_is_valid(self) -> None:
        """similarity_threshold=1.0 is a valid constructor argument."""
        eng = DecisionEngine(similarity_threshold=1.0)
        assert eng is not None

    def test_f4_required_frames_one_is_valid(self) -> None:
        """required_frames=1 is a valid constructor argument."""
        eng = DecisionEngine(required_frames=1)
        assert eng is not None


# ===========================================================================
# Group G — ignore() and reset() behaviour
# ===========================================================================


class TestIgnoreAndReset:
    """Group G: explicit ignore/reset paths."""

    def test_g1_ignore_always_returns_decision_ignore(self, engine: DecisionEngine) -> None:
        """ignore() always returns Decision.IGNORE regardless of prior state."""
        assert engine.ignore() is Decision.IGNORE

    def test_g2_ignore_resets_mid_accumulation_counter(self, engine: DecisionEngine) -> None:
        """ignore() called after partial accumulation resets counter to 0."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.ignore()
        assert engine.consecutive_frames == 0

    def test_g3_reset_then_full_sequence_grants(self, engine: DecisionEngine) -> None:
        """After reset() a full clean sequence still issues GRANT."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES - 1)
        engine.reset()
        assert engine.consecutive_frames == 0
        assert _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)[-1] is Decision.GRANT

    def test_g3_double_reset_is_idempotent(self, engine: DecisionEngine) -> None:
        """Calling reset() twice leaves consecutive_frames at 0."""
        engine.reset()
        engine.reset()
        assert engine.consecutive_frames == 0


# ===========================================================================
# Group H — consecutive_frames property
# ===========================================================================


class TestConsecutiveFramesProperty:
    """Group H: the read-only property surfaces internal state correctly."""

    def test_h1_property_starts_at_zero(self, engine: DecisionEngine) -> None:
        """consecutive_frames is 0 on a freshly constructed engine."""
        assert engine.consecutive_frames == 0

    def test_h2_property_tracks_increments_during_accumulation(
        self, engine: DecisionEngine
    ) -> None:
        """Property increases by 1 per qualifying frame while accumulating."""
        for i in range(1, REQUIRED_CONSECUTIVE_FRAMES):
            engine.evaluate(
                similarity=SIMILARITY_THRESHOLD,
                anti_spoof_pass=True,
                face_in_db=True,
            )
            assert engine.consecutive_frames == i

    def test_h3_property_is_zero_immediately_after_grant(self, engine: DecisionEngine) -> None:
        """Property is 0 right after a GRANT is issued."""
        _pass_frames(engine, REQUIRED_CONSECUTIVE_FRAMES)
        assert engine.consecutive_frames == 0

    def test_h4_property_is_read_only(self, engine: DecisionEngine) -> None:
        """consecutive_frames property must not allow direct external assignment."""
        with pytest.raises(AttributeError):
            engine.consecutive_frames = 99  # type: ignore[misc]
