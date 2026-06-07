#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_decision_to_actuator.py — IT-1.

Integration test: DecisionEngine ↔ ActuatorController contract.

Unit tests 已驗證的邊界
-----------------------
  test_decision_engine.py   : DecisionEngine.evaluate() 的每個分支（純邏輯，無硬體）
  test_actuator_controller.py : ActuatorController 各方法（注入 MagicMock hardware）
  test_orchestrator.py       : Orchestrator._act() 的分派（actuator 是 MagicMock）

IT-1 填補的空缺
---------------
上面三組測試都在各自的邊界 mock 掉對方。IT-1 把 **真實** DecisionEngine 和
**真實** ActuatorController 同時接起來，透過 Orchestrator._tick() 驅動，
確認：

  1. DecisionEngine.evaluate() 對特定輸入回傳正確的 Decision enum。
  2. Orchestrator._act() 把該 Decision 分派到正確的 actuator 方法。
  3. 真實 ActuatorController 內部對 LED / Buzzer / Servo 的呼叫序列正確。

若任一層的介面改變（例如 kwargs 改名、呼叫順序錯誤），IT-1 在這裡攔截，
而 unit tests 因為各自只測一層所以不會發現。

Mock 邊界
---------
  真實執行：DecisionEngine、ActuatorController、Orchestrator._tick() / _act()
  Mock：     LED.indicate()、Buzzer._beep()、Servo.unlock_then_relock()  ← GPIO 邊界
             FaceDetector.detect()、FaceRecognizer.match()、AntiSpoof.predict()
             MqttPublisher、HCSR04._measure_distance()

CI 相容性
---------
不依賴真實 GPIO，可在 ubuntu-latest runner 上執行。
整合測試放在 tests/integration/，CI 的 unit-test job 以 --ignore=tests/integration
跳過本目錄；本目錄由 Jetson self-hosted runner 上的 integration-test job 執行。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.actuator_controller import (
    _ALERT_BEEPS,
    _BEEP_ON_S,
    _DENY_LED_S,
    _GRANT_LED_S,
    ActuatorController,
)
from src.decision_engine import DecisionEngine
from src.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# 共用的假資料建構函式
# （刻意與 test_orchestrator.py 保持相同命名，方便日後閱讀對照）
# ---------------------------------------------------------------------------


def _blank() -> np.ndarray:
    """空白 480×640 BGR 影像，供 _tick() 當作相機 frame 使用。."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_face() -> MagicMock:
    """
    模擬 FaceDetector.detect() 回傳的單一人臉物件。.

    bbox 必須是 numpy array，因為 _tick() 中執行：
        bbox = face.bbox.astype(int).tolist()
    若用普通 MagicMock 而不指定 .bbox，astype() 會丟 AttributeError。
    """
    face = MagicMock()
    face.bbox = np.array([10, 20, 110, 120], dtype=np.float32)
    face.crop = np.zeros((100, 100, 3), dtype=np.uint8)
    return face


def _make_recog(
    name: str = "alice",
    similarity: float = 0.92,
    authorized: bool = True,
) -> MagicMock:
    """
    模擬 FaceRecognizer.match() 的回傳值。.

    authorized 對應 _tick() 中的 face_in_db：
        face_in_db = recog.authorized
    """
    r = MagicMock()
    r.name = name
    r.similarity = similarity
    r.authorized = authorized
    return r


def _make_liveness(is_live: bool = True, score: float = 0.88) -> MagicMock:
    """
    模擬 AntiSpoof.predict() 的回傳值。.

    is_live 對應 _tick() 中的 anti_spoof_pass：
        anti_spoof_pass = liveness.is_live
    """
    lv = MagicMock()
    lv.is_live = is_live
    lv.score = score
    return lv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hw() -> dict:
    """
    GPIO 邊界 mock：LED、Buzzer、Servo 各一個 MagicMock。.

    這是 IT-1 的 mock 邊界。ActuatorController 透過依賴注入接收這三個物件，
    所以真實的 GPIO 程式碼不會被執行，測試可以在 x86 CI 上跑。
    """
    return {
        "led": MagicMock(),
        "buzzer": MagicMock(),
        "servo": MagicMock(),
    }


@pytest.fixture()
def engine() -> DecisionEngine:
    """真實 DecisionEngine，使用 proposal §4.5 的生產參數。."""
    return DecisionEngine(similarity_threshold=0.85, required_frames=3)


@pytest.fixture()
def actuator(hw: dict) -> ActuatorController:
    """真實 ActuatorController，注入 mock GPIO hardware。."""
    return ActuatorController(
        led=hw["led"],
        buzzer=hw["buzzer"],
        servo=hw["servo"],
    )


@pytest.fixture()
def ai_mocks() -> dict:
    """AI pipeline 的三個 mock：detector、recognizer、antispoof。."""
    return {
        "detector": MagicMock(),
        "recognizer": MagicMock(),
        "antispoof": MagicMock(),
    }


@pytest.fixture()
def orc(engine: DecisionEngine, actuator: ActuatorController, ai_mocks: dict) -> Orchestrator:
    """完整接線的 Orchestrator：真實 engine + 真實 actuator + mock AI/sensor/publisher."""
    return Orchestrator(
        detector=ai_mocks["detector"],
        recognizer=ai_mocks["recognizer"],
        antispoof=ai_mocks["antispoof"],
        engine=engine,
        actuator=actuator,
        publisher=MagicMock(),
        sensor=MagicMock(),
        display=False,
    )


# ---------------------------------------------------------------------------
# 驅動 _tick() 的輔助函式
# ---------------------------------------------------------------------------


def _run_tick(
    orc: Orchestrator,
    ai_mocks: dict,
    recog: MagicMock,
    liveness: MagicMock,
    distance: float = 30.0,
) -> None:
    """
    執行一次 Orchestrator._tick()。.

    - sensor._measure_distance 回傳 distance（預設 30 cm，小於門檻 60 cm → gate open）
    - detector 回傳一張人臉
    - recognizer / antispoof 回傳指定的 recog / liveness 物件
    """
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = recog
    ai_mocks["antispoof"].predict.return_value = liveness
    with patch.object(orc._sensor, "_measure_distance", return_value=distance):
        orc._tick(_blank())


# ---------------------------------------------------------------------------
# IT-1-A：GRANT 路徑
#
# 驗證點：三幀累積後，真實 DecisionEngine 輸出 GRANT，
#         Orchestrator._act() 分派到 ActuatorController.grant_access()，
#         真實 grant_access() 內部呼叫 servo.unlock_then_relock 和 LED.indicate(success=True)。
#
# Unit tests 的空缺：test_orchestrator.py 的 test_grant_calls_grant_access_actuator
# 只驗證 mock_actuator.grant_access 被呼叫，無法保證真實 grant_access() 內部
# 對 servo 和 LED 的呼叫序列正確。
# ---------------------------------------------------------------------------


class TestGrantPath:
    """三幀累積 → GRANT → servo 解鎖 + 綠燈。."""

    @pytest.fixture(autouse=True)
    def _drive_grant(self, orc: Orchestrator, ai_mocks: dict, hw: dict) -> None:
        """每個測試前共同執行：送三幀匹配訊號，等 grant thread 完成。."""
        recog = _make_recog(similarity=0.92, authorized=True)
        liveness = _make_liveness(is_live=True)
        for _ in range(3):
            _run_tick(orc, ai_mocks, recog, liveness)
        # grant_access() 在 daemon thread 中執行。
        # servo/LED 都是 mock（instant），0.1 s 足以讓 thread 排程並完成。
        time.sleep(0.1)

    def test_servo_unlock_is_called(self, hw: dict) -> None:
        hw["servo"].unlock_then_relock.assert_called_once()

    def test_green_led_is_called_with_correct_duration(self, hw: dict) -> None:
        hw["led"].indicate.assert_called_once_with(success=True, duration=_GRANT_LED_S)

    def test_buzzer_is_silent_on_grant(self, hw: dict) -> None:
        hw["buzzer"]._beep.assert_not_called()

    def test_two_frames_do_not_trigger_grant(
        self,
        engine: DecisionEngine,
        actuator: ActuatorController,
        ai_mocks: dict,
    ) -> None:
        """
        二幀不足以觸發 GRANT（engine 仍在累積中）。.

        這個測試使用獨立的 engine 和 actuator（不用 _drive_grant fixture）
        以免被前三幀的狀態污染。
        """
        fresh_hw = {"led": MagicMock(), "buzzer": MagicMock(), "servo": MagicMock()}
        fresh_ctrl = ActuatorController(
            led=fresh_hw["led"],
            buzzer=fresh_hw["buzzer"],
            servo=fresh_hw["servo"],
        )
        fresh_engine = DecisionEngine(similarity_threshold=0.85, required_frames=3)
        fresh_orc = Orchestrator(
            detector=ai_mocks["detector"],
            recognizer=ai_mocks["recognizer"],
            antispoof=ai_mocks["antispoof"],
            engine=fresh_engine,
            actuator=fresh_ctrl,
            publisher=MagicMock(),
            sensor=MagicMock(),
            display=False,
        )
        recog = _make_recog(similarity=0.92, authorized=True)
        liveness = _make_liveness(is_live=True)
        for _ in range(2):  # 只送兩幀
            _run_tick(fresh_orc, ai_mocks, recog, liveness)
        time.sleep(0.1)

        fresh_hw["servo"].unlock_then_relock.assert_not_called()
        fresh_hw["led"].indicate.assert_not_called()


# ---------------------------------------------------------------------------
# IT-1-B：DENY 路徑
#
# 條件：anti_spoof_pass=True、face_in_db=True（authorized=True）、
#        similarity < 0.85 → DecisionEngine 輸出 DENY。
#
# 驗證點：真實 deny_access() 只亮紅燈，不響 buzzer、不動 servo。
# ---------------------------------------------------------------------------


class TestDenyPath:
    """低相似度（臉在 DB 但不夠像） → DENY → 紅燈、靜音、servo 不動。."""

    @pytest.fixture(autouse=True)
    def _drive_deny(self, orc: Orchestrator, ai_mocks: dict) -> None:
        recog = _make_recog(
            similarity=0.70,  # 低於門檻 0.85
            authorized=True,  # face_in_db=True，所以不會觸發 UNKNOWN
        )
        liveness = _make_liveness(is_live=True)
        _run_tick(orc, ai_mocks, recog, liveness)
        time.sleep(0.1)  # deny_access() mock LED → instant，0.1 s 足夠

    def test_red_led_is_called_with_correct_duration(self, hw: dict) -> None:
        hw["led"].indicate.assert_called_once_with(success=False, duration=_DENY_LED_S)

    def test_buzzer_is_silent_on_deny(self, hw: dict) -> None:
        hw["buzzer"]._beep.assert_not_called()

    def test_servo_is_not_moved_on_deny(self, hw: dict) -> None:
        hw["servo"].unlock_then_relock.assert_not_called()


# ---------------------------------------------------------------------------
# IT-1-C：UNKNOWN 路徑
#
# 條件：anti_spoof_pass=True、face_in_db=False（authorized=False）→ UNKNOWN。
#
# 驗證點：
#   1. buzzer._beep 被呼叫恰好 _ALERT_BEEPS（3）次。
#   2. 每次呼叫的 duration 都是 _BEEP_ON_S（0.20 s）。
#   3. 紅燈亮起（LED.indicate(success=False, duration=_DENY_LED_S)）。
#   4. servo 不動。
#
# Unit tests 空缺：test_actuator_controller.py 已驗證 ActuatorController
# 在拿到 mock buzzer 時呼叫 _beep 三次，但沒有驗證這個呼叫是從 UNKNOWN
# 決策路徑正確分派過來的。IT-1-C 補足這個垂直整合。
#
# 等待策略：
#   alert_unknown() 在 daemon=False thread 中執行（修正後）。
#   _multi_beep(3) 內部有 2 × _BEEP_OFF_S（0.15 s）的真實 sleep，
#   所以 thread 至少需要 0.30 s 才能完成。等 0.5 s 留有充裕餘量。
# ---------------------------------------------------------------------------


class TestUnknownPath:
    """臉不在 DB → UNKNOWN → 紅燈 + 三聲 buzzer。."""

    @pytest.fixture(autouse=True)
    def _drive_unknown(self, orc: Orchestrator, ai_mocks: dict) -> None:
        recog = _make_recog(
            similarity=0.50,
            authorized=False,  # face_in_db=False → UNKNOWN
        )
        liveness = _make_liveness(is_live=True)
        _run_tick(orc, ai_mocks, recog, liveness)
        # _multi_beep(3) 有 2 × 0.15 s = 0.30 s 真實 sleep，等 0.5 s。
        time.sleep(0.5)

    def test_buzzer_beeps_exactly_alert_beeps_times(self, hw: dict) -> None:
        assert hw["buzzer"]._beep.call_count == _ALERT_BEEPS

    def test_each_beep_uses_correct_duration(self, hw: dict) -> None:
        for c in hw["buzzer"]._beep.call_args_list:
            assert c == call(_BEEP_ON_S)

    def test_red_led_is_called(self, hw: dict) -> None:
        hw["led"].indicate.assert_called_once_with(success=False, duration=_DENY_LED_S)

    def test_servo_is_not_moved_on_unknown(self, hw: dict) -> None:
        hw["servo"].unlock_then_relock.assert_not_called()


# ---------------------------------------------------------------------------
# IT-1-D：SPOOF 路徑
#
# 條件：anti_spoof_pass=False（liveness.is_live=False）→ SPOOF。
#        DecisionEngine 優先級 1：spoof 優先於 UNKNOWN、DENY。
#
# 驗證點：硬體行為與 UNKNOWN 完全相同（紅燈 + 三聲），
#          但上游的決策路徑不同（anti_spoof 失敗）。
#
# 為何要獨立測試（而非只測 UNKNOWN）：
#   CAPSTONE 要求兩者觸發不同的 MQTT payload（decision="SPOOF" vs "UNKNOWN"），
#   但硬體行為相同。若把 SPOOF 分派到錯誤的 actuator 方法，MQTT 和硬體都會出錯。
# ---------------------------------------------------------------------------


class TestSpoofPath:
    """活體偵測失敗 → SPOOF → 紅燈 + 三聲 buzzer（硬體同 UNKNOWN，決策路徑不同）。."""

    @pytest.fixture(autouse=True)
    def _drive_spoof(self, orc: Orchestrator, ai_mocks: dict) -> None:
        recog = _make_recog(
            similarity=0.92,
            authorized=True,  # 臉在 DB，但活體失敗，優先輸出 SPOOF
        )
        liveness = _make_liveness(is_live=False, score=0.20)  # 活體失敗
        _run_tick(orc, ai_mocks, recog, liveness)
        time.sleep(0.5)

    def test_buzzer_beeps_exactly_alert_beeps_times(self, hw: dict) -> None:
        assert hw["buzzer"]._beep.call_count == _ALERT_BEEPS

    def test_each_beep_uses_correct_duration(self, hw: dict) -> None:
        for c in hw["buzzer"]._beep.call_args_list:
            assert c == call(_BEEP_ON_S)

    def test_red_led_is_called(self, hw: dict) -> None:
        hw["led"].indicate.assert_called_once_with(success=False, duration=_DENY_LED_S)

    def test_servo_is_not_moved_on_spoof(self, hw: dict) -> None:
        hw["servo"].unlock_then_relock.assert_not_called()


# ---------------------------------------------------------------------------
# IT-1-E：IGNORE 路徑（累積中的第一幀）
#
# 條件：所有條件符合 GRANT，但幀數不足（第 1 幀，需要 3 幀）→ IGNORE。
#
# 驗證點：在 IGNORE 狀態，任何 actuator 方法都不應被呼叫。
#
# 這個測試看似簡單，但它保護了一個重要不變量：
# Orchestrator._act(Decision.IGNORE) 必須完全靜默。
# 若未來有人在 _act() 裡不小心為 IGNORE 加了 actuator 呼叫，這個測試會攔截。
# ---------------------------------------------------------------------------


class TestIgnorePath:
    """累積中的第一幀 → IGNORE → 所有 actuator 靜默。."""

    @pytest.fixture(autouse=True)
    def _drive_ignore(self, orc: Orchestrator, ai_mocks: dict) -> None:
        recog = _make_recog(similarity=0.92, authorized=True)
        liveness = _make_liveness(is_live=True)
        _run_tick(orc, ai_mocks, recog, liveness)  # 第 1 幀 → IGNORE
        time.sleep(0.1)

    def test_servo_is_not_called(self, hw: dict) -> None:
        hw["servo"].unlock_then_relock.assert_not_called()

    def test_led_is_not_called(self, hw: dict) -> None:
        hw["led"].indicate.assert_not_called()

    def test_buzzer_is_not_called(self, hw: dict) -> None:
        hw["buzzer"]._beep.assert_not_called()
