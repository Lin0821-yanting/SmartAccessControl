#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/test_orchestrator.py — unit tests for Orchestrator.

Strategy
--------
* ALL hardware dependencies (FaceDetector, FaceRecognizer, AntiSpoof,
  ActuatorController, MqttPublisher, HCSR04) are replaced with MagicMocks.
* DecisionEngine is used as the *real* class (pure logic, no hardware).
* run() and from_config() are excluded via pragma: no cover — they require
  a live camera and physical hardware that cannot exist in CI.
* _tick() is called directly to exercise every code path without a camera.

Coverage targets
----------------
  _tick             — gate closed / cooldown / no-face / all 5 decisions
  _act              — GRANT / DENY / UNKNOWN / SPOOF / IGNORE
  _set_door_state   — state-change publish / no-op when state unchanged
  _publish_event_from — delegates correctly to publisher
  _update_fps       — sub-1s window / flush after 1s
  _set_stage / _get_stage — thread-safe round-trip
  _heartbeat_loop   — one iteration, exception-swallowing branch
  _read_cpu_temp    — happy path + OSError fallback
  _read_ram_gb      — happy path + OSError fallback

Run with:
    pytest tests/test_orchestrator.py -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Real classes used in tests
from src.decision_engine import Decision, DecisionEngine

# Module under test
from src.orchestrator import Orchestrator, _read_cpu_temp, _read_ram_gb


# ---------------------------------------------------------------------------
# Helpers for building mock AI outputs
# ---------------------------------------------------------------------------

def _make_face(bbox=(10, 20, 110, 120), conf=0.95):
    """Return a mock FaceDetection with a valid crop array."""
    face = MagicMock()
    face.bbox = np.array(bbox, dtype=np.float32)
    face.confidence = conf
    face.crop = np.zeros((100, 100, 3), dtype=np.uint8)
    return face


def _make_recog(name="alice", similarity=0.92, authorized=True):
    r = MagicMock()
    r.name = name
    r.similarity = similarity
    r.authorized = authorized
    return r


def _make_liveness(is_live=True, score=0.88):
    lv = MagicMock()
    lv.is_live = is_live
    lv.score = score
    return lv


# ---------------------------------------------------------------------------
# Fixture: a fully-wired Orchestrator with all hardware mocked
# ---------------------------------------------------------------------------

@pytest.fixture()
def mocks():
    """Return a dict of all injected mock objects."""
    return {
        "detector":   MagicMock(),
        "recognizer": MagicMock(),
        "antispoof":  MagicMock(),
        "actuator":   MagicMock(),
        "publisher":  MagicMock(),
        "sensor":     MagicMock(),
    }


@pytest.fixture()
def engine():
    """Real DecisionEngine (pure logic — no hardware)."""
    return DecisionEngine(similarity_threshold=0.85, required_frames=3)


@pytest.fixture()
def orc(mocks, engine):
    """Orchestrator with all hardware replaced by mocks."""
    return Orchestrator(
        detector=mocks["detector"],
        recognizer=mocks["recognizer"],
        antispoof=mocks["antispoof"],
        engine=engine,
        actuator=mocks["actuator"],
        publisher=mocks["publisher"],
        sensor=mocks["sensor"],
        display=False,
    )


# ---------------------------------------------------------------------------
# __init__ state
# ---------------------------------------------------------------------------

class TestInit:
    def test_initial_door_state_is_locked(self, orc):
        assert orc._door_state == "locked"

    def test_initial_last_person_is_unknown(self, orc):
        assert orc._last_person == "unknown"

    def test_initial_pipeline_stage_is_idle(self, orc):
        assert orc._get_stage() == "IDLE"

    def test_initial_fps_is_zero(self, orc):
        assert orc._fps == 0.0


# ---------------------------------------------------------------------------
# _tick — HC-SR04 gate path
# ---------------------------------------------------------------------------

class TestTickGate:
    def _blank_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_gate_closed_returns_early_no_detection(self, orc, mocks):
        """When distance >= 60 cm, detector must NOT be called."""
        mocks["sensor"].measure_distance.return_value = 80.0
        orc._tick(self._blank_frame())
        mocks["detector"].detect.assert_not_called()

    def test_gate_none_distance_returns_early(self, orc, mocks):
        """None from HC-SR04 (timeout) must also skip AI pipeline."""
        mocks["sensor"].measure_distance.return_value = None
        orc._tick(self._blank_frame())
        mocks["detector"].detect.assert_not_called()

    def test_cooldown_suppresses_pipeline(self, orc, mocks):
        """During grant cooldown window, AI pipeline must be skipped."""
        mocks["sensor"].measure_distance.return_value = 30.0
        orc._grant_until = time.monotonic() + 10.0   # force active cooldown
        orc._tick(self._blank_frame())
        mocks["detector"].detect.assert_not_called()

    def test_gate_open_calls_detector(self, orc, mocks):
        """When distance < 60 cm and not in cooldown, detector must be called."""
        mocks["sensor"].measure_distance.return_value = 40.0
        mocks["detector"].detect.return_value = []    # no faces
        orc._tick(self._blank_frame())
        mocks["detector"].detect.assert_called_once()


# ---------------------------------------------------------------------------
# _tick — no-face path
# ---------------------------------------------------------------------------

class TestTickNoFace:
    def _open_gate(self, mocks):
        mocks["sensor"].measure_distance.return_value = 40.0
        mocks["detector"].detect.return_value = []

    def test_no_face_publishes_ignore_event(self, orc, mocks):
        self._open_gate(mocks)
        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))
        mocks["publisher"].publish_event.assert_called_once()
        kwargs = mocks["publisher"].publish_event.call_args.kwargs
        assert kwargs["decision"] == "IGNORE"

    def test_no_face_event_has_unknown_identity(self, orc, mocks):
        self._open_gate(mocks)
        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))
        kwargs = mocks["publisher"].publish_event.call_args.kwargs
        assert kwargs["identity"] == "unknown"
        assert kwargs["bbox"] is None

    def test_no_face_does_not_call_actuator(self, orc, mocks):
        self._open_gate(mocks)
        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))
        mocks["actuator"].grant_access.assert_not_called()
        mocks["actuator"].deny_access.assert_not_called()


# ---------------------------------------------------------------------------
# _tick — full pipeline: GRANT path (3 consecutive frames)
# ---------------------------------------------------------------------------

class TestTickGrant:
    def _setup(self, mocks, recog=None, liveness=None):
        mocks["sensor"].measure_distance.return_value = 30.0
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = recog or _make_recog()
        mocks["antispoof"].predict.return_value = liveness or _make_liveness()

    def test_grant_fires_on_third_consecutive_frame(self, orc, mocks):
        self._setup(mocks)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orc._tick(frame)  # frame 1 → IGNORE (accumulating)
        orc._tick(frame)  # frame 2 → IGNORE (accumulating)
        orc._tick(frame)  # frame 3 → GRANT

        # publisher.publish_event was called 3 times; the last one is GRANT
        calls = mocks["publisher"].publish_event.call_args_list
        assert calls[-1].kwargs["decision"] == "GRANT"

    def test_grant_calls_grant_access_actuator(self, orc, mocks):
        self._setup(mocks)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            orc._tick(frame)
        # give daemon thread a moment to execute
        time.sleep(0.05)
        mocks["actuator"].grant_access.assert_called_once()

    def test_grant_publishes_status_unlocked(self, orc, mocks):
        self._setup(mocks)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            orc._tick(frame)
        # allow daemon thread + timer setup
        time.sleep(0.05)
        status_calls = [
            c for c in mocks["publisher"].publish_status.call_args_list
            if c.kwargs.get("door_state") == "unlocked"
        ]
        assert len(status_calls) >= 1

    def test_grant_event_carries_correct_identity(self, orc, mocks):
        self._setup(mocks, recog=_make_recog(name="bob", similarity=0.93))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            orc._tick(frame)
        last_kwargs = mocks["publisher"].publish_event.call_args_list[-1].kwargs
        assert last_kwargs["identity"] == "bob"


# ---------------------------------------------------------------------------
# _tick — full pipeline: DENY path
# ---------------------------------------------------------------------------

class TestTickDeny:
    def test_deny_calls_deny_access(self, orc, mocks):
        mocks["sensor"].measure_distance.return_value = 30.0
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog(
            name="unknown", similarity=0.70, authorized=False
        )
        # face_in_db=False triggers UNKNOWN, so use authorized=True + low sim
        mocks["recognizer"].match.return_value = _make_recog(
            name="alice", similarity=0.70, authorized=True
        )
        # DecisionEngine: authorized=True but sim < 0.85 → DENY
        # We need to force face_in_db=True but similarity<threshold
        # recognizer.authorized is used as face_in_db, so set authorized=True
        # but similarity < threshold
        mocks["antispoof"].predict.return_value = _make_liveness()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        orc._tick(frame)

        time.sleep(0.05)
        last_event = mocks["publisher"].publish_event.call_args.kwargs
        assert last_event["decision"] == "DENY"
        mocks["actuator"].deny_access.assert_called_once()


# ---------------------------------------------------------------------------
# _tick — full pipeline: UNKNOWN path
# ---------------------------------------------------------------------------

class TestTickUnknown:
    def test_unknown_calls_alert_unknown(self, orc, mocks):
        mocks["sensor"].measure_distance.return_value = 30.0
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog(
            name="unknown", similarity=0.50, authorized=False
        )
        mocks["antispoof"].predict.return_value = _make_liveness()

        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))
        time.sleep(0.05)

        last_event = mocks["publisher"].publish_event.call_args.kwargs
        assert last_event["decision"] == "UNKNOWN"
        mocks["actuator"].alert_unknown.assert_called_once()


# ---------------------------------------------------------------------------
# _tick — full pipeline: SPOOF path
# ---------------------------------------------------------------------------

class TestTickSpoof:
    def test_spoof_calls_alert_spoof(self, orc, mocks):
        mocks["sensor"].measure_distance.return_value = 30.0
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog()
        mocks["antispoof"].predict.return_value = _make_liveness(is_live=False, score=0.20)

        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))
        time.sleep(0.05)

        last_event = mocks["publisher"].publish_event.call_args.kwargs
        assert last_event["decision"] == "SPOOF"
        mocks["actuator"].alert_spoof.assert_called_once()

    def test_spoof_event_carries_spoof_score(self, orc, mocks):
        mocks["sensor"].measure_distance.return_value = 30.0
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog()
        mocks["antispoof"].predict.return_value = _make_liveness(is_live=False, score=0.15)

        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))
        kwargs = mocks["publisher"].publish_event.call_args.kwargs
        assert kwargs["spoof_score"] == pytest.approx(0.15, abs=1e-4)
        assert kwargs["is_live"] is False


# ---------------------------------------------------------------------------
# _tick — IGNORE (accumulating) path: no actuator fired
# ---------------------------------------------------------------------------

class TestTickIgnoreAccumulating:
    def test_accumulating_frame_fires_no_actuator(self, orc, mocks):
        """First frame toward GRANT: engine is accumulating → IGNORE returned."""
        mocks["sensor"].measure_distance.return_value = 30.0
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog()
        mocks["antispoof"].predict.return_value = _make_liveness()

        orc._tick(np.zeros((480, 640, 3), dtype=np.uint8))  # frame 1 → IGNORE
        time.sleep(0.05)

        mocks["actuator"].grant_access.assert_not_called()
        mocks["actuator"].deny_access.assert_not_called()
        mocks["actuator"].alert_unknown.assert_not_called()
        mocks["actuator"].alert_spoof.assert_not_called()


# ---------------------------------------------------------------------------
# _act — each decision branch in isolation
# ---------------------------------------------------------------------------

class TestAct:
    def test_act_grant_calls_grant_access_and_unlock(self, orc, mocks):
        orc._act(Decision.GRANT, identity="alice")
        time.sleep(0.05)
        mocks["actuator"].grant_access.assert_called_once()

    def test_act_deny_calls_deny_access(self, orc, mocks):
        orc._act(Decision.DENY, identity="unknown")
        time.sleep(0.05)
        mocks["actuator"].deny_access.assert_called_once()

    def test_act_unknown_calls_alert_unknown(self, orc, mocks):
        orc._act(Decision.UNKNOWN, identity="unknown")
        time.sleep(0.05)
        mocks["actuator"].alert_unknown.assert_called_once()

    def test_act_spoof_calls_alert_spoof(self, orc, mocks):
        orc._act(Decision.SPOOF, identity="unknown")
        time.sleep(0.05)
        mocks["actuator"].alert_spoof.assert_called_once()

    def test_act_ignore_calls_no_actuator(self, orc, mocks):
        orc._act(Decision.IGNORE, identity="unknown")
        time.sleep(0.05)
        mocks["actuator"].grant_access.assert_not_called()
        mocks["actuator"].deny_access.assert_not_called()
        mocks["actuator"].alert_unknown.assert_not_called()
        mocks["actuator"].alert_spoof.assert_not_called()


# ---------------------------------------------------------------------------
# _set_door_state — publish-only-on-change logic
# ---------------------------------------------------------------------------

class TestSetDoorState:
    def test_state_change_triggers_publish(self, orc, mocks):
        orc._door_state = "locked"
        orc._set_door_state("unlocked", "alice")
        mocks["publisher"].publish_status.assert_called_once_with(
            door_state="unlocked", last_person="alice"
        )

    def test_same_state_does_not_publish(self, orc, mocks):
        orc._door_state = "locked"
        orc._set_door_state("locked", "alice")
        mocks["publisher"].publish_status.assert_not_called()

    def test_state_change_updates_last_person(self, orc, mocks):
        orc._set_door_state("unlocked", "bob")
        assert orc._last_person == "bob"

    def test_consecutive_different_states_publish_twice(self, orc, mocks):
        orc._set_door_state("unlocked", "alice")
        orc._set_door_state("locked", "alice")
        assert mocks["publisher"].publish_status.call_count == 2


# ---------------------------------------------------------------------------
# _update_fps
# ---------------------------------------------------------------------------

class TestUpdateFps:
    def test_fps_not_flushed_within_window(self, orc):
        orc._fps_window_start = time.monotonic()  # reset window
        orc._fps = 0.0
        orc._frame_count = 0
        orc._update_fps()   # only 1 call, < 1s elapsed
        assert orc._fps == 0.0          # not flushed yet
        assert orc._frame_count == 1

    def test_fps_flushed_after_window(self, orc):
        # Force the window start into the past so elapsed >= 1s
        orc._fps_window_start = time.monotonic() - 2.0
        orc._frame_count = 10
        orc._fps = 0.0
        orc._update_fps()   # triggers flush
        assert orc._fps > 0.0
        assert orc._frame_count == 0    # reset after flush


# ---------------------------------------------------------------------------
# _set_stage / _get_stage — thread safety
# ---------------------------------------------------------------------------

class TestStage:
    def test_set_and_get_stage_roundtrip(self, orc):
        orc._set_stage("MATCHING")
        assert orc._get_stage() == "MATCHING"

    def test_stage_transitions(self, orc):
        for stage in ("IDLE", "DETECTING", "MATCHING", "DECIDED"):
            orc._set_stage(stage)
            assert orc._get_stage() == stage

    def test_concurrent_stage_writes_do_not_corrupt(self, orc):
        """Light concurrency smoke test — no assertion error expected."""
        results = []

        def writer(stage):
            orc._set_stage(stage)
            results.append(orc._get_stage())

        threads = [threading.Thread(target=writer, args=(s,))
                   for s in ("IDLE", "DETECTING", "MATCHING", "DECIDED") * 5]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All values must be valid stage strings
        valid = {"IDLE", "DETECTING", "MATCHING", "DECIDED"}
        assert all(r in valid for r in results)


# ---------------------------------------------------------------------------
# _heartbeat_loop — one-iteration test (stop after first publish)
# ---------------------------------------------------------------------------

class TestHeartbeatLoop:
    def test_heartbeat_calls_publish_heartbeat(self, orc, mocks):
        """Fire one heartbeat iteration and verify publish_heartbeat is called."""
        call_count = []

        def fake_sleep(sec):
            # Let the loop run once, then raise SystemExit to break the while True
            if len(call_count) >= 1:
                raise SystemExit
            call_count.append(1)

        with patch("src.orchestrator.time.sleep", side_effect=fake_sleep), \
             patch("src.orchestrator._read_cpu_temp", return_value=55.0), \
             patch("src.orchestrator._read_ram_gb", return_value=2.5):
            try:
                orc._heartbeat_loop()
            except SystemExit:
                pass

        mocks["publisher"].publish_heartbeat.assert_called_once()

    def test_heartbeat_swallows_publish_exception(self, orc, mocks):
        """Exception inside publish_heartbeat must not crash the loop."""
        call_count = []

        def fake_sleep(sec):
            if len(call_count) >= 1:
                raise SystemExit
            call_count.append(1)

        mocks["publisher"].publish_heartbeat.side_effect = RuntimeError("broker down")

        with patch("src.orchestrator.time.sleep", side_effect=fake_sleep):
            try:
                orc._heartbeat_loop()
            except SystemExit:
                pass
        # No exception propagated — test passes if we reach here


# ---------------------------------------------------------------------------
# _read_cpu_temp / _read_ram_gb — module-level helpers
# ---------------------------------------------------------------------------

class TestSystemMetrics:
    def test_read_cpu_temp_happy_path(self, tmp_path):
        fake = tmp_path / "temp"
        fake.write_text("52000\n")
        with patch("src.orchestrator.Path", return_value=fake):
            result = _read_cpu_temp()
        assert result == pytest.approx(52.0)

    def test_read_cpu_temp_oserror_returns_minus_one(self):
        with patch("src.orchestrator.Path") as mock_path:
            mock_path.return_value.read_text.side_effect = OSError
            result = _read_cpu_temp()
        assert result == -1.0

    def test_read_ram_gb_happy_path(self, tmp_path):
        fake = tmp_path / "meminfo"
        # 8 GB total, 4 GB available → 4 GB used
        fake.write_text(
            "MemTotal:       8388608 kB\n"
            "MemFree:        2097152 kB\n"
            "MemAvailable:   4194304 kB\n"
        )
        with patch("src.orchestrator.Path", return_value=fake):
            result = _read_ram_gb()
        assert result == pytest.approx(4.0, abs=0.01)

    def test_read_ram_gb_oserror_returns_minus_one(self):
        with patch("src.orchestrator.Path") as mock_path:
            mock_path.return_value.read_text.side_effect = OSError
            result = _read_ram_gb()
        assert result == -1.0
