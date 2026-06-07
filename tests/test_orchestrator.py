#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/test_orchestrator.py — unit tests for Orchestrator.

Strategy
--------
* ALL hardware dependencies are replaced with MagicMocks.
* Sensor method is patched via patch.object to guarantee correct return values.
* DecisionEngine is used as the real class (pure logic, no hardware).
* run() and from_config() are excluded via pragma: no cover.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.decision_engine import Decision, DecisionEngine
from src.orchestrator import Orchestrator, _read_cpu_temp, _read_ram_gb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_face(bbox=(10, 20, 110, 120), conf=0.95):
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mocks():
    """All injected mock objects."""
    return {
        "detector": MagicMock(),
        "recognizer": MagicMock(),
        "antispoof": MagicMock(),
        "actuator": MagicMock(),
        "publisher": MagicMock(),
        "sensor": MagicMock(),
    }


@pytest.fixture()
def engine():
    return DecisionEngine(similarity_threshold=0.85, required_frames=3)


@pytest.fixture()
def orc(mocks, engine):
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


def _blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


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
# _tick — gate path
# ---------------------------------------------------------------------------


class TestTickGate:
    def test_gate_closed_timeout_skips_pipeline(self, orc, mocks):
        """Inf distance (timeout) must skip AI pipeline."""
        with patch.object(orc._sensor, "_measure_distance", return_value=float("inf")):
            orc._tick(_blank())
        mocks["detector"].detect.assert_not_called()

    def test_gate_closed_large_distance_skips_pipeline(self, orc, mocks):
        """Distance >= 60 cm must skip AI pipeline."""
        with patch.object(orc._sensor, "_measure_distance", return_value=80.0):
            orc._tick(_blank())
        mocks["detector"].detect.assert_not_called()

    def test_cooldown_suppresses_pipeline(self, orc, mocks):
        """During grant cooldown, AI pipeline must be skipped."""
        orc._grant_until = time.monotonic() + 10.0
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())
        mocks["detector"].detect.assert_not_called()

    def test_gate_open_calls_detector(self, orc, mocks):
        """Distance < 60 cm with no cooldown must call detector."""
        mocks["detector"].detect.return_value = []
        with patch.object(orc._sensor, "_measure_distance", return_value=40.0):
            orc._tick(_blank())
        mocks["detector"].detect.assert_called_once()


# ---------------------------------------------------------------------------
# _tick — no-face path
# ---------------------------------------------------------------------------


class TestTickNoFace:
    def _tick_open(self, orc, mocks):
        mocks["detector"].detect.return_value = []
        with patch.object(orc._sensor, "_measure_distance", return_value=40.0):
            orc._tick(_blank())

    def test_no_face_publishes_ignore_event(self, orc, mocks):
        self._tick_open(orc, mocks)
        mocks["publisher"].publish_event.assert_called_once()
        assert mocks["publisher"].publish_event.call_args.kwargs["decision"] == "IGNORE"

    def test_no_face_event_has_unknown_identity(self, orc, mocks):
        self._tick_open(orc, mocks)
        kwargs = mocks["publisher"].publish_event.call_args.kwargs
        assert kwargs["identity"] == "unknown"
        assert kwargs["bbox"] is None

    def test_no_face_does_not_call_actuator(self, orc, mocks):
        self._tick_open(orc, mocks)
        mocks["actuator"].grant_access.assert_not_called()
        mocks["actuator"].deny_access.assert_not_called()


# ---------------------------------------------------------------------------
# _tick — GRANT path (3 consecutive frames)
# ---------------------------------------------------------------------------


class TestTickGrant:
    def _tick_matching(self, orc, mocks, recog=None, liveness=None):
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = recog or _make_recog()
        mocks["antispoof"].predict.return_value = liveness or _make_liveness()
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())

    def test_grant_fires_on_third_consecutive_frame(self, orc, mocks):
        for _ in range(3):
            self._tick_matching(orc, mocks)
        calls = mocks["publisher"].publish_event.call_args_list
        assert calls[-1].kwargs["decision"] == "GRANT"

    def test_grant_calls_grant_access_actuator(self, orc, mocks):
        for _ in range(3):
            self._tick_matching(orc, mocks)
        time.sleep(0.05)
        mocks["actuator"].grant_access.assert_called_once()

    def test_grant_publishes_status_unlocked(self, orc, mocks):
        for _ in range(3):
            self._tick_matching(orc, mocks)
        time.sleep(0.05)
        unlocked = [
            c
            for c in mocks["publisher"].publish_status.call_args_list
            if c.kwargs.get("door_state") == "unlocked"
        ]
        assert len(unlocked) >= 1

    def test_grant_event_carries_correct_identity(self, orc, mocks):
        for _ in range(3):
            self._tick_matching(orc, mocks, recog=_make_recog(name="bob", similarity=0.93))
        assert mocks["publisher"].publish_event.call_args_list[-1].kwargs["identity"] == "bob"


# ---------------------------------------------------------------------------
# _tick — DENY path
# ---------------------------------------------------------------------------


class TestTickDeny:
    def test_deny_calls_deny_access(self, orc, mocks):
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog(
            name="alice", similarity=0.70, authorized=True
        )
        mocks["antispoof"].predict.return_value = _make_liveness()
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())
        time.sleep(0.05)
        assert mocks["publisher"].publish_event.call_args.kwargs["decision"] == "DENY"
        mocks["actuator"].deny_access.assert_called_once()


# ---------------------------------------------------------------------------
# _tick — UNKNOWN path
# ---------------------------------------------------------------------------


class TestTickUnknown:
    def test_unknown_calls_alert_unknown(self, orc, mocks):
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog(
            name="unknown", similarity=0.50, authorized=False
        )
        mocks["antispoof"].predict.return_value = _make_liveness()
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())
        time.sleep(0.05)
        assert mocks["publisher"].publish_event.call_args.kwargs["decision"] == "UNKNOWN"
        mocks["actuator"].alert_unknown.assert_called_once()


# ---------------------------------------------------------------------------
# _tick — SPOOF path
# ---------------------------------------------------------------------------


class TestTickSpoof:
    def test_spoof_calls_alert_spoof(self, orc, mocks):
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog()
        mocks["antispoof"].predict.return_value = _make_liveness(is_live=False, score=0.20)
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())
        time.sleep(0.05)
        assert mocks["publisher"].publish_event.call_args.kwargs["decision"] == "SPOOF"
        mocks["actuator"].alert_spoof.assert_called_once()

    def test_spoof_event_carries_spoof_score(self, orc, mocks):
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog()
        mocks["antispoof"].predict.return_value = _make_liveness(is_live=False, score=0.15)
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())
        kwargs = mocks["publisher"].publish_event.call_args.kwargs
        assert kwargs["spoof_score"] == pytest.approx(0.15, abs=1e-4)
        assert kwargs["is_live"] is False


# ---------------------------------------------------------------------------
# _tick — IGNORE (accumulating)
# ---------------------------------------------------------------------------


class TestTickIgnoreAccumulating:
    def test_accumulating_frame_fires_no_actuator(self, orc, mocks):
        mocks["detector"].detect.return_value = [_make_face()]
        mocks["recognizer"].match.return_value = _make_recog()
        mocks["antispoof"].predict.return_value = _make_liveness()
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())  # frame 1 → IGNORE
        time.sleep(0.05)
        mocks["actuator"].grant_access.assert_not_called()
        mocks["actuator"].deny_access.assert_not_called()
        mocks["actuator"].alert_unknown.assert_not_called()
        mocks["actuator"].alert_spoof.assert_not_called()


# ---------------------------------------------------------------------------
# _act — each decision
# ---------------------------------------------------------------------------


class TestAct:
    def test_act_grant_calls_grant_access(self, orc, mocks):
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
# _set_door_state
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
        orc._fps_window_start = time.monotonic()
        orc._fps = 0.0
        orc._frame_count = 0
        orc._update_fps()
        assert orc._fps == 0.0
        assert orc._frame_count == 1

    def test_fps_flushed_after_window(self, orc):
        orc._fps_window_start = time.monotonic() - 2.0
        orc._frame_count = 10
        orc._fps = 0.0
        orc._update_fps()
        assert orc._fps > 0.0
        assert orc._frame_count == 0


# ---------------------------------------------------------------------------
# _set_stage / _get_stage
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
        results = []

        def writer(stage):
            orc._set_stage(stage)
            results.append(orc._get_stage())

        threads = [
            threading.Thread(target=writer, args=(s,))
            for s in ("IDLE", "DETECTING", "MATCHING", "DECIDED") * 5
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        valid = {"IDLE", "DETECTING", "MATCHING", "DECIDED"}
        assert all(r in valid for r in results)


# ---------------------------------------------------------------------------
# _heartbeat_loop
# ---------------------------------------------------------------------------


class TestHeartbeatLoop:
    def test_heartbeat_calls_publish_heartbeat(self, orc, mocks):
        call_count = []

        def fake_sleep(sec):
            if len(call_count) >= 1:
                raise SystemExit
            call_count.append(1)

        with (
            patch("src.orchestrator.time.sleep", side_effect=fake_sleep),
            patch("src.orchestrator._read_cpu_temp", return_value=55.0),
            patch("src.orchestrator._read_ram_gb", return_value=2.5),
        ):
            try:
                orc._heartbeat_loop()
            except SystemExit:
                pass

        mocks["publisher"].publish_heartbeat.assert_called_once()

    def test_heartbeat_swallows_publish_exception(self, orc, mocks):
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


# ---------------------------------------------------------------------------
# _read_cpu_temp / _read_ram_gb
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
        fake.write_text(
            "MemTotal:       8388608 kB\nMemFree:        2097152 kB\nMemAvailable:   4194304 kB\n"
        )
        with patch("src.orchestrator.Path", return_value=fake):
            result = _read_ram_gb()
        assert result == pytest.approx(4.0, abs=0.01)

    def test_read_ram_gb_oserror_returns_minus_one(self):
        with patch("src.orchestrator.Path") as mock_path:
            mock_path.return_value.read_text.side_effect = OSError
            result = _read_ram_gb()
        assert result == -1.0
