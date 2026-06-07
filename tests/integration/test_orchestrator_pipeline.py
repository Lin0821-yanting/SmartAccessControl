#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_orchestrator_pipeline.py — IT-3

Integration test: Orchestrator 跨幀狀態機行為。

IT-3 與現有測試的分工
--------------------
  test_orchestrator.py  : 單次 _tick() 行為（每個 class 對應一個決策路徑）
  IT-1                  : 每個決策的硬體序列（GRANT→servo+LED，UNKNOWN→3 beeps 等）
  IT-2                  : Buzzer GPIO 電位序列與方法完成時間
  IT-3（本檔）          : **多幀跨 tick** 的狀態機正確性

IT-3 填補的空缺
---------------
test_orchestrator.py 和 IT-1 都只測「單次 _tick()」的結果。以下行為需要送
多幀才能觀察，因此在現有測試中完全缺席：

  A. GRANT 後 consecutive_frames 重置 → 需再累積 3 幀才能第二次 GRANT。
  B. DENY / UNKNOWN / SPOOF 打斷累積後，後續幀必須從 0 重新計數。
  C. Cooldown 由真實 GRANT _tick() 自動觸發（非手動設定 _grant_until）。
     且 cooldown 到期後偵測重新開放。
  D. Auto-relock threading.Timer 到期後 publish_status("locked") 被呼叫。
  E. MQTT publish_event 的 consecutive_frames 欄位在多幀中正確遞增。

Mock 邊界
---------
  真實執行：DecisionEngine、Orchestrator._tick() / _act() / _set_door_state()
  Mock：     ActuatorController（所有方法，使 thread 即時完成）
             MqttPublisher（驗證 MQTT 呼叫序列）
             FaceDetector、FaceRecognizer、AntiSpoof、HCSR04

ActuatorController 使用 MagicMock 而非真實實例，原因：
  IT-3 的焦點是跨幀狀態機，不是硬體序列（硬體序列由 IT-1/IT-2 負責）。
  MagicMock actuator 讓 grant_access 等方法即時完成，不需等待 thread，
  使多幀序列測試保持快速且確定性。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.decision_engine import DecisionEngine
from src.orchestrator import _GRANT_COOLDOWN_S, Orchestrator

# ---------------------------------------------------------------------------
# 假資料建構函式（與 IT-1 保持相同命名與邏輯）
# ---------------------------------------------------------------------------


def _blank() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_face() -> MagicMock:
    face = MagicMock()
    face.bbox = np.array([10, 20, 110, 120], dtype=np.float32)
    face.crop = np.zeros((100, 100, 3), dtype=np.uint8)
    return face


def _make_recog(
    name: str = "alice",
    similarity: float = 0.92,
    authorized: bool = True,
) -> MagicMock:
    r = MagicMock()
    r.name = name
    r.similarity = similarity
    r.authorized = authorized
    return r


def _make_liveness(is_live: bool = True, score: float = 0.88) -> MagicMock:
    lv = MagicMock()
    lv.is_live = is_live
    lv.score = score
    return lv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> DecisionEngine:
    """真實 DecisionEngine，使用生產參數（threshold=0.85，required=3）。"""
    return DecisionEngine(similarity_threshold=0.85, required_frames=3)


@pytest.fixture()
def ai_mocks() -> dict:
    return {
        "detector": MagicMock(),
        "recognizer": MagicMock(),
        "antispoof": MagicMock(),
    }


@pytest.fixture()
def publisher() -> MagicMock:
    """Mock MqttPublisher，用於驗證 MQTT 呼叫序列。"""
    return MagicMock()


@pytest.fixture()
def actuator() -> MagicMock:
    """Mock ActuatorController — 讓 grant_access 等方法即時完成。"""
    return MagicMock()


@pytest.fixture()
def orc(
    engine: DecisionEngine, actuator: MagicMock, ai_mocks: dict, publisher: MagicMock
) -> Orchestrator:
    """Orchestrator：真實 engine，其餘全 mock。"""
    return Orchestrator(
        detector=ai_mocks["detector"],
        recognizer=ai_mocks["recognizer"],
        antispoof=ai_mocks["antispoof"],
        engine=engine,
        actuator=actuator,
        publisher=publisher,
        sensor=MagicMock(),
        display=False,
    )


# ---------------------------------------------------------------------------
# 驅動 _tick() 的輔助函式
# ---------------------------------------------------------------------------


def _tick_matching(
    orc: Orchestrator,
    ai_mocks: dict,
    name: str = "alice",
    similarity: float = 0.92,
) -> None:
    """送出一幀「完全符合條件」的訊號（high similarity, live, in DB）。"""
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog(
        name=name, similarity=similarity, authorized=True
    )
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=True)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_deny(orc: Orchestrator, ai_mocks: dict) -> None:
    """送出一幀 DENY 訊號（similarity < threshold，但 face in DB）。"""
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog(similarity=0.70, authorized=True)
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=True)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_unknown(orc: Orchestrator, ai_mocks: dict) -> None:
    """送出一幀 UNKNOWN 訊號（face not in DB）。"""
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog(similarity=0.50, authorized=False)
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=True)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_spoof(orc: Orchestrator, ai_mocks: dict) -> None:
    """送出一幀 SPOOF 訊號（liveness 失敗）。"""
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog()
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=False, score=0.20)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _send_grant(orc: Orchestrator, ai_mocks: dict, name: str = "alice") -> None:
    """送 3 幀讓 engine 到達 GRANT，並等 actuator thread 完成。"""
    for _ in range(3):
        _tick_matching(orc, ai_mocks, name=name)
    time.sleep(0.05)  # actuator mock 即時，0.05 s 足以讓 thread 排程完成


# ---------------------------------------------------------------------------
# IT-3-A：GRANT 後幀計數器重置
#
# 這是現有測試中完全缺席的行為：第一次 GRANT 後，DecisionEngine 的
# consecutive_frames 重置為 0，後續幀必須重新累積 3 次才能第二次 GRANT。
# ---------------------------------------------------------------------------


class TestFrameCounterResetAfterGrant:
    """GRANT 後計數器歸零 → 需再累積 3 幀。"""

    def test_engine_counter_is_zero_immediately_after_grant(
        self, orc: Orchestrator, ai_mocks: dict, engine: DecisionEngine
    ) -> None:
        """
        第 3 幀後 engine.consecutive_frames 必須是 0。

        DecisionEngine.evaluate() 在 GRANT 後呼叫 _reset()，
        這個測試確認 Orchestrator 正確驅動了這個重置。
        """
        _send_grant(orc, ai_mocks)
        assert engine.consecutive_frames == 0

    def test_fourth_frame_after_grant_does_not_trigger_second_grant(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """
        GRANT 冷卻期間第 4 幀被 cooldown 攔截，不會呼叫 grant_access。

        此測試驗證 cooldown 機制由真實 GRANT 流程（非手動設定 _grant_until）
        自動觸發，並確實阻擋後續觸發。
        """
        _send_grant(orc, ai_mocks)
        actuator.grant_access.reset_mock()  # 清除第一次 GRANT 的記錄

        _tick_matching(orc, ai_mocks)  # 第 4 幀（在冷卻中）
        time.sleep(0.05)

        actuator.grant_access.assert_not_called()

    def test_second_grant_fires_on_sixth_matching_frame(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """
        第 1-3 幀 → GRANT（第一次），冷卻過後第 4-6 幀 → GRANT（第二次）。

        測試方式：手動設定 _grant_until=0 讓冷卻立刻結束，
        然後送 3 幀確認第二次 GRANT 觸發。
        """
        _send_grant(orc, ai_mocks)

        orc._grant_until = 0.0  # 讓冷卻立即到期
        actuator.grant_access.reset_mock()

        for _ in range(3):
            _tick_matching(orc, ai_mocks)  # 第 4、5、6 幀
        time.sleep(0.05)

        actuator.grant_access.assert_called_once()

    def test_two_frames_after_cooldown_expiry_do_not_retrigger(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """
        冷卻到期後只送 2 幀，不應觸發 GRANT（需要 3 幀）。
        """
        _send_grant(orc, ai_mocks)
        orc._grant_until = 0.0
        actuator.grant_access.reset_mock()

        for _ in range(2):  # 只送 2 幀
            _tick_matching(orc, ai_mocks)
        time.sleep(0.05)

        actuator.grant_access.assert_not_called()


# ---------------------------------------------------------------------------
# IT-3-B：中斷累積 → 後續幀需從 0 重新計數
#
# 場景：2 幀匹配後出現 1 幀 DENY/UNKNOWN/SPOOF，
#       接著再送 2 幀匹配 → 仍然是 IGNORE（計數器已被重置）。
#
# 現有測試中：
#   test_decision_engine.py 的 E2/E3/E4 測試了 DecisionEngine 本身的重置。
#   但沒有測試透過完整 _tick() 管線（sensor → AI → engine → act）的重置。
# ---------------------------------------------------------------------------


class TestInterruptedAccumulation:
    """中斷累積後需重新計數 3 幀。"""

    def test_deny_interrupts_accumulation(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """
        2 幀匹配 → 1 幀 DENY → 2 幀匹配 → IGNORE（共 5 幀，未達第 6 幀的 GRANT）。
        """
        for _ in range(2):
            _tick_matching(orc, ai_mocks)  # 幀 1、2：累積中
        _tick_deny(orc, ai_mocks)  # 幀 3：DENY → counter 重置為 0
        for _ in range(2):
            _tick_matching(orc, ai_mocks)  # 幀 4、5：重新開始，但只有 2 幀
        time.sleep(0.05)

        actuator.grant_access.assert_not_called()

    def test_unknown_interrupts_accumulation(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """2 幀匹配 → 1 幀 UNKNOWN → 2 幀匹配 → 仍 IGNORE。"""
        for _ in range(2):
            _tick_matching(orc, ai_mocks)
        _tick_unknown(orc, ai_mocks)  # counter 重置
        for _ in range(2):
            _tick_matching(orc, ai_mocks)
        time.sleep(0.05)

        actuator.grant_access.assert_not_called()

    def test_spoof_interrupts_accumulation(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """2 幀匹配 → 1 幀 SPOOF → 2 幀匹配 → 仍 IGNORE。"""
        for _ in range(2):
            _tick_matching(orc, ai_mocks)
        _tick_spoof(orc, ai_mocks)  # counter 重置
        for _ in range(2):
            _tick_matching(orc, ai_mocks)
        time.sleep(0.05)

        actuator.grant_access.assert_not_called()

    def test_three_frames_after_interruption_do_grant(
        self, orc: Orchestrator, ai_mocks: dict, actuator: MagicMock
    ) -> None:
        """
        中斷後再送 3 幀，第 3 幀仍可觸發 GRANT（驗證重新累積有效）。
        """
        _tick_matching(orc, ai_mocks)  # 幀 1：累積 1
        _tick_deny(orc, ai_mocks)  # 幀 2：DENY，counter → 0
        for _ in range(3):
            _tick_matching(orc, ai_mocks)  # 幀 3、4、5：重新累積 1、2、GRANT
        time.sleep(0.05)

        actuator.grant_access.assert_called_once()


# ---------------------------------------------------------------------------
# IT-3-C：Cooldown 由真實 GRANT 流程自動觸發
#
# test_orchestrator.py 的 test_cooldown_suppresses_pipeline 是手動設定
# orc._grant_until = time.monotonic() + 10.0，不走真實的 GRANT 路徑。
# IT-3-C 透過完整 3 幀 → GRANT 流程，驗證 cooldown 被正確設置和套用。
# ---------------------------------------------------------------------------


class TestCooldownMechanism:
    """Cooldown 由真實 GRANT 流程自動設定並生效。"""

    def test_grant_tick_sets_grant_until_in_future(self, orc: Orchestrator, ai_mocks: dict) -> None:
        """
        3 幀 → GRANT 後，orc._grant_until 必須大於當前時間。

        這確認 _act(Decision.GRANT) 確實執行了：
            self._grant_until = time.monotonic() + _GRANT_COOLDOWN_S
        """
        before = time.monotonic()
        _send_grant(orc, ai_mocks)
        assert orc._grant_until > before
        assert orc._grant_until == pytest.approx(before + _GRANT_COOLDOWN_S, abs=0.5)

    def test_cooldown_blocks_detector_on_immediate_next_tick(
        self, orc: Orchestrator, ai_mocks: dict
    ) -> None:
        """
        GRANT 後立即的第 4 幀 → detector 不被呼叫（cooldown 中）。

        這測試了完整的自動觸發路徑，而非手動設定 _grant_until。
        """
        _send_grant(orc, ai_mocks)
        ai_mocks["detector"].detect.reset_mock()

        # 第 4 幀：距離 < 60 cm，但 cooldown 尚未結束
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())

        ai_mocks["detector"].detect.assert_not_called()

    def test_expired_cooldown_allows_detection(self, orc: Orchestrator, ai_mocks: dict) -> None:
        """
        GRANT 後強制讓 cooldown 到期，下一幀 detector 應被呼叫。
        """
        _send_grant(orc, ai_mocks)
        orc._grant_until = 0.0  # 強制 cooldown 到期
        ai_mocks["detector"].detect.reset_mock()
        ai_mocks["detector"].detect.return_value = []

        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())

        ai_mocks["detector"].detect.assert_called_once()


# ---------------------------------------------------------------------------
# IT-3-D：Auto-relock threading.Timer 觸發後 publish_status("locked")
#
# _act(Decision.GRANT) 除了立即 publish_status("unlocked") 之外，
# 還啟動了一個 threading.Timer(_GRANT_COOLDOWN_S, self._set_door_state, ("locked", name))。
# 這個 Timer 在 _GRANT_COOLDOWN_S 秒後觸發，發布 publish_status("locked")。
#
# 測試方式：patch 常數為 0.05 s，讓 Timer 快速觸發，不需等 4 秒。
# ---------------------------------------------------------------------------


class TestAutoRelockTimer:
    """Auto-relock Timer 到期後發布 locked 狀態。"""

    _SHORT_COOLDOWN_S = 0.05

    def test_door_state_is_unlocked_after_grant(self, orc: Orchestrator, ai_mocks: dict) -> None:
        """
        3 幀 → GRANT 後，orc._door_state 應該是 "unlocked"。
        _set_door_state("unlocked", identity) 在 _act() 中同步執行。
        """
        _send_grant(orc, ai_mocks)
        assert orc._door_state == "unlocked"

    def test_auto_relock_timer_publishes_locked(self, ai_mocks: dict, publisher: MagicMock) -> None:
        """
        _GRANT_COOLDOWN_S 到期後，threading.Timer 觸發，
        publish_status(door_state="locked") 被呼叫。

        patch _GRANT_COOLDOWN_S = 0.05 讓測試在 < 1 s 內完成。
        """
        with patch("src.orchestrator._GRANT_COOLDOWN_S", self._SHORT_COOLDOWN_S):
            short_orc = Orchestrator(
                detector=ai_mocks["detector"],
                recognizer=ai_mocks["recognizer"],
                antispoof=ai_mocks["antispoof"],
                engine=DecisionEngine(similarity_threshold=0.85, required_frames=3),
                actuator=MagicMock(),
                publisher=publisher,
                sensor=MagicMock(),
                display=False,
            )
            for _ in range(3):
                _tick_matching(short_orc, ai_mocks)

        # 等 Timer 觸發（0.05 s + 排程餘量）
        time.sleep(self._SHORT_COOLDOWN_S * 4)

        locked_calls = [
            c
            for c in publisher.publish_status.call_args_list
            if c.kwargs.get("door_state") == "locked"
        ]
        assert len(locked_calls) >= 1

    def test_publish_status_unlocked_called_before_locked(
        self, ai_mocks: dict, publisher: MagicMock
    ) -> None:
        """
        狀態轉換順序必須是 unlocked → locked（不能反向）。
        """
        with patch("src.orchestrator._GRANT_COOLDOWN_S", self._SHORT_COOLDOWN_S):
            short_orc = Orchestrator(
                detector=ai_mocks["detector"],
                recognizer=ai_mocks["recognizer"],
                antispoof=ai_mocks["antispoof"],
                engine=DecisionEngine(similarity_threshold=0.85, required_frames=3),
                actuator=MagicMock(),
                publisher=publisher,
                sensor=MagicMock(),
                display=False,
            )
            for _ in range(3):
                _tick_matching(short_orc, ai_mocks)

        time.sleep(self._SHORT_COOLDOWN_S * 4)

        status_calls = [c.kwargs.get("door_state") for c in publisher.publish_status.call_args_list]
        # 必須包含 unlocked，且 unlocked 在 locked 之前
        assert "unlocked" in status_calls
        assert "locked" in status_calls
        assert status_calls.index("unlocked") < status_calls.index("locked")


# ---------------------------------------------------------------------------
# IT-3-E：MQTT publish_event 的 consecutive_frames 欄位跨幀遞增
#
# Orchestrator._publish_event_from() 把 engine.consecutive_frames 傳入
# publish_event()。這個欄位應該隨幀遞增，並在 GRANT 後歸零。
#
# 現有測試：無任何測試驗證 consecutive_frames 的跨幀行為。
# ---------------------------------------------------------------------------


class TestMqttEventSequence:
    """publish_event 的 consecutive_frames 欄位反映真實計數。"""

    def test_consecutive_frames_increments_across_ticks(
        self, orc: Orchestrator, ai_mocks: dict, publisher: MagicMock
    ) -> None:
        """
        幀 1 的 consecutive_frames = 1，幀 2 = 2。

        說明：DecisionEngine 在 IGNORE 路徑（累積中）不重置計數，
        所以呼叫 publish_event 時的 engine.consecutive_frames 反映真實累積。
        """
        for _ in range(2):
            _tick_matching(orc, ai_mocks)

        events = publisher.publish_event.call_args_list
        assert events[0].kwargs["consecutive_frames"] == 1
        assert events[1].kwargs["consecutive_frames"] == 2

    def test_grant_event_consecutive_frames_is_zero(
        self, orc: Orchestrator, ai_mocks: dict, publisher: MagicMock
    ) -> None:
        """
        第 3 幀（GRANT）的 consecutive_frames = 0。

        DecisionEngine._reset() 在 evaluate() 回傳 GRANT 時就執行，
        所以 publish_event_from() 讀到的 engine.consecutive_frames 已是 0。
        """
        for _ in range(3):
            _tick_matching(orc, ai_mocks)

        grant_event = publisher.publish_event.call_args_list[-1]
        assert grant_event.kwargs["decision"] == "GRANT"
        assert grant_event.kwargs["consecutive_frames"] == 0

    def test_deny_event_has_correct_decision_field(
        self, orc: Orchestrator, ai_mocks: dict, publisher: MagicMock
    ) -> None:
        """2 幀累積後送 1 幀 DENY → 最後一個 event 的 decision = 'DENY'。"""
        for _ in range(2):
            _tick_matching(orc, ai_mocks)
        _tick_deny(orc, ai_mocks)

        last_event = publisher.publish_event.call_args_list[-1]
        assert last_event.kwargs["decision"] == "DENY"
