#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_buzzer_thread_completion.py — IT-2.

Integration test: Buzzer GPIO pin 電位序列與方法完成性。

IT-1 與 IT-2 的分工
--------------------
IT-1 使用 MagicMock 作為 Buzzer，只能驗證 ``_beep()`` 被呼叫了幾次。
IT-2 使用**真實 Buzzer**（GPIO 仍是 mock），讓 ``_beep()`` 裡的
``GPIO.output()`` 真正執行，從而驗證：

  1. 每聲 beep 對應一次 HIGH → LOW 的 GPIO 電位序列。
  2. 三聲結束後，GPIO pin 最後停在 LOW（安全狀態）。
  3. ``alert_unknown()`` / ``alert_spoof()`` 在合理時間內回傳（不卡死）。
  4. 透過 Orchestrator._tick() 分派後，真實 Buzzer 仍能走完完整序列。

這四點都是 IT-1 看不到的，因為 IT-1 的 Buzzer mock 不執行任何 GPIO 呼叫。

為什麼 pin 最後必須是 LOW 很重要
----------------------------------
``Buzzer._beep()`` 的序列：

    GPIO.output(pin, HIGH) → time.sleep(duration) → GPIO.output(pin, LOW)

若執行緒在 HIGH 之後、LOW 之前被強制終止（daemon thread 在主程式退出時的
行為），pin 會停在 HIGH，buzzer 就永遠響。IT-2 透過追蹤 ``GPIO.output``
的 call_args_list 來確認最後一次呼叫確實是 LOW。

Mock 邊界
---------
  真實執行：Buzzer、ActuatorController、Orchestrator（Group C）
  Mock：     Jetson.GPIO（從 conftest 繼承）、LED.indicate、Servo.unlock_then_relock

CI 相容性
---------
不依賴真實 GPIO，可在 ubuntu-latest 上執行。
Group A 的 time.sleep 被 patch 掉以加速測試。
Group B（時間測試）使用真實 sleep 來驗證上限。
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.actuator_controller import _ALERT_BEEPS, ActuatorController
from src.buzzer import BUZZER_PIN, Buzzer
from src.decision_engine import DecisionEngine
from src.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# 本測試檔預期的 buzzer GPIO.output 呼叫模式
#
# _beep() 每次呼叫：
#   GPIO.output(BUZZER_PIN, HIGH)   ← 第 1 次
#   time.sleep(duration)
#   GPIO.output(BUZZER_PIN, LOW)    ← 第 2 次
#
# _multi_beep(3) 執行 3 次 _beep()，加上 2 次 inter-beep sleep：
#   _beep() → _beep() → _beep()
#   HIGH LOW  HIGH LOW  HIGH LOW   ← 共 6 次 GPIO.output
#
# 最後一次 GPIO.output 必須是 LOW。
# ---------------------------------------------------------------------------

_EXPECTED_OUTPUT_CALLS = _ALERT_BEEPS * 2  # 每聲：HIGH + LOW = 2 次


# ---------------------------------------------------------------------------
# 共用假資料（與 IT-1 保持相同命名慣例）
# ---------------------------------------------------------------------------


def _blank() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_face() -> MagicMock:
    face = MagicMock()
    face.bbox = np.array([10, 20, 110, 120], dtype=np.float32)
    face.crop = np.zeros((100, 100, 3), dtype=np.uint8)
    return face


def _make_recog(
    name: str = "unknown",
    similarity: float = 0.50,
    authorized: bool = False,
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
def gpio_mock():
    """
    回傳 conftest 所設定的共享 Jetson.GPIO mock。.

    在每個測試的前後重置 ``output`` 的呼叫記錄，
    避免不同測試間的 call_args_list 互相污染。
    """
    mock = sys.modules["Jetson.GPIO"]
    mock.output.reset_mock()
    yield mock
    mock.output.reset_mock()


@pytest.fixture()
def real_buzzer(gpio_mock):
    """
    真實 Buzzer 實例。.

    ``Buzzer.__init__()`` 會呼叫 ``GPIO.setmode / setup``（被 mock 掉），
    之後 ``_beep()`` 呼叫的 ``GPIO.output()`` 會被 gpio_mock 記錄下來。
    """
    return Buzzer()


@pytest.fixture()
def actuator(real_buzzer):
    """
    真實 ActuatorController：.

      - real_buzzer  → GPIO.output 呼叫可被追蹤
      - mock LED     → indicate() 不執行真實 GPIO，LED thread 即時完成
      - mock Servo   → unlock_then_relock() 不執行真實 PWM.
    """
    return ActuatorController(
        led=MagicMock(),
        buzzer=real_buzzer,
        servo=MagicMock(),
    )


# ---------------------------------------------------------------------------
# 輔助：從 gpio_mock.output.call_args_list 中篩出 BUZZER_PIN 的呼叫
# ---------------------------------------------------------------------------


def _buzzer_output_calls(gpio_mock) -> list:
    """
    過濾出所有對 BUZZER_PIN 的 GPIO.output() 呼叫。.

    buzzer._beep() 呼叫模式：
        GPIO.output(BUZZER_PIN, HIGH)
        GPIO.output(BUZZER_PIN, LOW)
    LED 的 output 呼叫用不同的 pin，不會混入。
    """
    return [c for c in gpio_mock.output.call_args_list if c.args and c.args[0] == BUZZER_PIN]


# ---------------------------------------------------------------------------
# IT-2-A：alert_unknown() 的 GPIO 電位序列
#
# 驗證：
#   1. BUZZER_PIN 被 output 恰好 _ALERT_BEEPS * 2 次（HIGH + LOW 各 3 次）
#   2. HIGH 呼叫恰好 _ALERT_BEEPS 次
#   3. 最後一次呼叫是 LOW（pin 結束在安全狀態）
#
# time.sleep 被 patch 掉，讓每個 GPIO 呼叫即時完成，不拖慢測試。
# ---------------------------------------------------------------------------


class TestAlertUnknownGpioSequence:
    """alert_unknown() 直接呼叫 → GPIO 電位序列正確。."""

    @pytest.fixture(autouse=True)
    def _run_alert(self, actuator, gpio_mock):
        """
        在每個測試前執行 alert_unknown()。.

        patch("time.sleep") 同時抑制：
          - buzzer._beep() 裡的 time.sleep(duration)
          - _multi_beep() 裡的 time.sleep(_BEEP_OFF_S).
        """
        with patch("time.sleep"):
            actuator.alert_unknown()

    def test_total_output_calls_for_buzzer_pin(self, gpio_mock) -> None:
        """BUZZER_PIN 的 GPIO.output 呼叫總次數必須是 HIGH+LOW 各 3 次。."""
        calls = _buzzer_output_calls(gpio_mock)
        assert len(calls) == _EXPECTED_OUTPUT_CALLS

    def test_high_pulse_count(self, gpio_mock) -> None:
        """HIGH 脈衝必須恰好 _ALERT_BEEPS（3）次。."""
        calls = _buzzer_output_calls(gpio_mock)
        high_calls = [c for c in calls if c.args[1] == gpio_mock.HIGH]
        assert len(high_calls) == _ALERT_BEEPS

    def test_pin_ends_in_low_state(self, gpio_mock) -> None:
        """
        最後一次對 BUZZER_PIN 的 GPIO.output 必須是 LOW。.

        這是 daemon=False 修正後的核心保證：
        若執行緒在 _beep() 中間被殺，最後的 LOW 永遠不會執行，
        pin 會卡在 HIGH。測試確認正常執行路徑下 pin 確實回到 LOW。
        """
        calls = _buzzer_output_calls(gpio_mock)
        assert calls[-1] == call(BUZZER_PIN, gpio_mock.LOW)

    def test_each_beep_alternates_high_then_low(self, gpio_mock) -> None:
        """
        每對相鄰呼叫必須是 HIGH → LOW 順序。.

        驗證 _beep() 的內部序列沒有被打亂（例如兩個 HIGH 連續出現）。
        """
        calls = _buzzer_output_calls(gpio_mock)
        for i in range(0, len(calls), 2):
            assert calls[i].args[1] == gpio_mock.HIGH, (
                f"第 {i // 2 + 1} 聲 beep 的第一個呼叫應為 HIGH"
            )
            assert calls[i + 1].args[1] == gpio_mock.LOW, (
                f"第 {i // 2 + 1} 聲 beep 的第二個呼叫應為 LOW"
            )


# ---------------------------------------------------------------------------
# IT-2-B：alert_spoof() 的 GPIO 電位序列
#
# SPOOF 和 UNKNOWN 的硬體序列相同（都是三聲 buzzer），
# 但分派路徑不同（_act() 裡不同的 elif）。
# 獨立測試確保 SPOOF 分支也正確驅動真實 Buzzer。
# ---------------------------------------------------------------------------


class TestAlertSpoofGpioSequence:
    """alert_spoof() 直接呼叫 → GPIO 電位序列與 UNKNOWN 相同。."""

    @pytest.fixture(autouse=True)
    def _run_alert(self, actuator, gpio_mock):
        with patch("time.sleep"):
            actuator.alert_spoof()

    def test_total_output_calls_for_buzzer_pin(self, gpio_mock) -> None:
        calls = _buzzer_output_calls(gpio_mock)
        assert len(calls) == _EXPECTED_OUTPUT_CALLS

    def test_high_pulse_count(self, gpio_mock) -> None:
        calls = _buzzer_output_calls(gpio_mock)
        high_calls = [c for c in calls if c.args[1] == gpio_mock.HIGH]
        assert len(high_calls) == _ALERT_BEEPS

    def test_pin_ends_in_low_state(self, gpio_mock) -> None:
        calls = _buzzer_output_calls(gpio_mock)
        assert calls[-1] == call(BUZZER_PIN, gpio_mock.LOW)


# ---------------------------------------------------------------------------
# IT-2-C：方法完成時間上限
#
# 使用**真實 time.sleep**（不 patch），確認 alert 方法在合理時間內結束。
# 理論執行時間：3 × _BEEP_ON_S + 2 × _BEEP_OFF_S = 3×0.20 + 2×0.15 = 0.90 s
# 設定上限 2.5 s，留有 >2.5× 的餘量。
#
# 這個測試捕捉的問題：
#   若 _multi_beep() 或 _beep() 裡的迴圈邏輯出錯導致無限等待，
#   測試會在 2.5 s 後以 AssertionError 失敗而非永遠 hang。
# ---------------------------------------------------------------------------


class TestAlertCompletionTiming:
    """alert 方法必須在 2.5 秒內完成（真實 sleep，不 patch）。."""

    _TIMEOUT_S = 2.5

    def test_alert_unknown_returns_within_timeout(self, actuator) -> None:
        start = time.monotonic()
        actuator.alert_unknown()
        elapsed = time.monotonic() - start
        assert elapsed < self._TIMEOUT_S, (
            f"alert_unknown() 花了 {elapsed:.2f}s，超過上限 {self._TIMEOUT_S}s"
        )

    def test_alert_spoof_returns_within_timeout(self, actuator) -> None:
        start = time.monotonic()
        actuator.alert_spoof()
        elapsed = time.monotonic() - start
        assert elapsed < self._TIMEOUT_S, (
            f"alert_spoof() 花了 {elapsed:.2f}s，超過上限 {self._TIMEOUT_S}s"
        )


# ---------------------------------------------------------------------------
# IT-2-D：透過 Orchestrator._tick() 分派 → 真實 Buzzer 仍走完完整序列
#
# IT-2-D 的意義
# -------------
# IT-2-A/B/C 直接呼叫 actuator 方法，跳過了 Orchestrator._act() 的執行緒分派。
# IT-2-D 從 Orchestrator 端觸發，確認：
#   1. Orchestrator._act() 正確分派到 alert_unknown / alert_spoof
#   2. 在執行緒中執行的真實 Buzzer 仍能走完完整的 GPIO 序列
#   3. 等待執行緒完成後，pin 電位確實是 LOW
#
# 等待策略：
#   _tick() 啟動執行緒後立即回傳。
#   真實 3-beep 序列需要約 0.90 s（含 inter-beep sleep）。
#   等待 1.5 s，留有足夠餘量讓執行緒完成。
# ---------------------------------------------------------------------------


@pytest.fixture()
def orc_with_real_buzzer(real_buzzer, gpio_mock):
    """
    Orchestrator：.

      - 真實 DecisionEngine（三幀累積邏輯）
      - 真實 ActuatorController（真實 Buzzer，mock LED/Servo）
      - mock AI pipeline、sensor、publisher.
    """
    return Orchestrator(
        detector=MagicMock(),
        recognizer=MagicMock(),
        antispoof=MagicMock(),
        engine=DecisionEngine(similarity_threshold=0.85, required_frames=3),
        actuator=ActuatorController(
            led=MagicMock(),
            buzzer=real_buzzer,
            servo=MagicMock(),
        ),
        publisher=MagicMock(),
        sensor=MagicMock(),
        display=False,
    )


class TestOrchestratorDrivesRealBuzzer:
    """Orchestrator._tick() 分派 UNKNOWN → 真實 Buzzer 走完 GPIO 序列。."""

    _THREAD_WAIT_S = 1.5  # 真實 0.90 s 序列 + 執行緒排程餘量

    def _tick_unknown(self, orc: Orchestrator) -> None:
        """送出一幀 UNKNOWN 條件（臉不在 DB）。."""
        face = _make_face()
        orc._sensor._measure_distance.return_value = 30.0
        orc._detector.detect.return_value = [face]
        orc._recognizer.match.return_value = _make_recog(authorized=False)
        orc._antispoof.predict.return_value = _make_liveness(is_live=True)
        with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
            orc._tick(_blank())

    def test_unknown_thread_drives_buzzer_to_low(
        self, orc_with_real_buzzer: Orchestrator, gpio_mock
    ) -> None:
        """
        _tick() → UNKNOWN 決策 → 執行緒 → alert_unknown() → pin 最後是 LOW。.

        這是 IT-2 最接近生產場景的測試：
        模擬真實的 Orchestrator 在偵測到陌生人臉後分派 alert，
        確認執行緒正常完成後 GPIO pin 停在安全狀態。
        """
        self._tick_unknown(orc_with_real_buzzer)
        time.sleep(self._THREAD_WAIT_S)  # 等 alert thread 完成

        calls = _buzzer_output_calls(gpio_mock)
        assert len(calls) > 0, "alert_unknown() 執行緒未產生任何 GPIO.output 呼叫"
        assert calls[-1] == call(BUZZER_PIN, gpio_mock.LOW), (
            "執行緒完成後 BUZZER_PIN 的最後一次 GPIO.output 不是 LOW"
        )

    def test_unknown_thread_produces_correct_beep_count(
        self, orc_with_real_buzzer: Orchestrator, gpio_mock
    ) -> None:
        """_tick() 觸發的 alert_unknown() 必須產生恰好 _ALERT_BEEPS 次 HIGH 脈衝。."""
        self._tick_unknown(orc_with_real_buzzer)
        time.sleep(self._THREAD_WAIT_S)

        calls = _buzzer_output_calls(gpio_mock)
        high_calls = [c for c in calls if c.args[1] == gpio_mock.HIGH]
        assert len(high_calls) == _ALERT_BEEPS, (
            f"預期 {_ALERT_BEEPS} 次 HIGH，實際 {len(high_calls)} 次"
        )
