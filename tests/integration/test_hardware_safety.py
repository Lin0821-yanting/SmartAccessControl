#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_hardware_safety.py — IT-5

Integration test: 硬體安全 — cleanup() 後所有 GPIO pin 回到 LOW。

與現有測試的分工
----------------
  test_sensors.py TestBuzzer.test_cleanup_drives_pin_low_and_releases
      → 測試 ``Buzzer.cleanup()`` 個別行為（reset_mock 在每個 test 前後）

  test_actuator_controller.py TestCleanup
      → 測試 ``actuator.cleanup()`` 有呼叫 mock_led.cleanup()、mock_buzzer.cleanup()
      → LED 和 Buzzer 都是 MagicMock，內部的 GPIO.output() 從未執行

  IT-2 TestAlertUnknownGpioSequence
      → 測試 ``_beep()`` 在**正常操作**結束後 pin 是 LOW（運行中的狀態）

IT-5 填補的空缺
---------------
IT-5 使用**真實 LED + 真實 Buzzer**（GPIO 仍是 mock）組入真實 ActuatorController，
驗證「shutdown 路徑」——即 ``cleanup()`` 被呼叫時：

  A. LED.cleanup()  → GPIO.output(green_pin, LOW) + GPIO.output(red_pin, LOW)
  B. Buzzer.cleanup() → GPIO.output(buzzer_pin, LOW)
  C. ActuatorController.cleanup() 正確傳播至所有硬體（LED + Buzzer 三個 pin 全 LOW）
  D. alert 執行完後呼叫 cleanup() 不拋例外（安全狀態）
  E. 重複呼叫 cleanup() 不拋例外（冪等性）

這是目前唯一能驗證「cleanup 後 GPIO 電位」的測試。
unit tests 和 IT-2 都看不到這個行為，因為它們要嘛把 LED/Buzzer mock 掉，
要嘛只看正常操作路徑的電位。

CAPSTONE demo 的硬體安全要求
-----------------------------
proposal §4.5 要求 Ctrl-C 後門鎖恢復到上鎖狀態、LED 熄滅、buzzer 靜音。
run() 的 finally 區塊呼叫 actuator.cleanup()，IT-5 確認這個 cleanup 確實
把所有 output pin 拉回 LOW。

Mock 邊界
---------
  真實執行：LED、Buzzer、ActuatorController.cleanup()
  Mock：     Jetson.GPIO（從 conftest 繼承）、Servo（gpiod，非 Jetson.GPIO）

CI 相容性
---------
不依賴真實 GPIO，可在 ubuntu-latest runner 上執行。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from src.actuator_controller import ActuatorController
from src.buzzer import BUZZER_PIN, Buzzer
from src.led import GREEN_LED_PIN, LED, RED_LED_PIN


# ---------------------------------------------------------------------------
# GPIO pin 常數（方便斷言）
# ---------------------------------------------------------------------------
_ALL_OUTPUT_PINS = (GREEN_LED_PIN, RED_LED_PIN, BUZZER_PIN)   # 7, 11, 29


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gpio_mock():
    """
    共享 Jetson.GPIO mock（由 conftest.py 設定）。
    在每個測試前後重置 output 和 cleanup 的呼叫記錄，
    避免跨測試的 call_args_list 互相污染。
    """
    mock = sys.modules["Jetson.GPIO"]
    mock.output.reset_mock()
    mock.cleanup.reset_mock()
    yield mock
    mock.output.reset_mock()
    mock.cleanup.reset_mock()


@pytest.fixture()
def led(gpio_mock):
    """
    真實 LED 實例（mocked GPIO）。

    建立後立即 reset GPIO.output 記錄，
    讓後續的測試只看到 cleanup() 相關的 GPIO 呼叫。
    """
    instance = LED()
    gpio_mock.output.reset_mock()   # 清除 __init__ 可能產生的呼叫
    return instance


@pytest.fixture()
def buzzer(gpio_mock):
    """真實 Buzzer 實例（mocked GPIO）。建立後重置 output 記錄。"""
    instance = Buzzer()
    gpio_mock.output.reset_mock()
    return instance


@pytest.fixture()
def actuator(led: LED, buzzer: Buzzer, gpio_mock) -> ActuatorController:
    """
    真實 ActuatorController：
      - 真實 LED（mock GPIO）→ cleanup() 會產生真實 GPIO.output(LOW) 呼叫
      - 真實 Buzzer（mock GPIO）→ 同上
      - mock Servo（gpiod，不是 Jetson.GPIO）→ 不影響 GPIO.output 記錄
    建立後重置 output 記錄，使測試只追蹤 cleanup 相關的 GPIO 呼叫。
    """
    ctrl = ActuatorController(
        led=led,
        buzzer=buzzer,
        servo=MagicMock(),
    )
    gpio_mock.output.reset_mock()
    return ctrl


# ---------------------------------------------------------------------------
# 輔助：從 GPIO.output 呼叫記錄中篩出指定 pin 被設為 LOW 的次數
# ---------------------------------------------------------------------------

def _low_calls_for_pin(gpio_mock, pin: int) -> list:
    """回傳所有 ``GPIO.output(pin, LOW)`` 的呼叫記錄。"""
    return [
        c for c in gpio_mock.output.call_args_list
        if c.args and c.args[0] == pin and c.args[1] == gpio_mock.LOW
    ]


def _pin_ends_low(gpio_mock, pin: int) -> bool:
    """確認指定 pin 的最後一次 GPIO.output 呼叫是 LOW。"""
    pin_calls = [c for c in gpio_mock.output.call_args_list if c.args and c.args[0] == pin]
    if not pin_calls:
        return False
    return pin_calls[-1].args[1] == gpio_mock.LOW


# ---------------------------------------------------------------------------
# IT-5-A：LED.cleanup() 的 GPIO 電位序列
#
# test_sensors.py 已測試 LED 個別行為，但：
#   1. 每個 test 用 autouse reset_mock，
#      無法測「在 ActuatorController 鏈中傳播後」的狀態。
#   2. 沒有明確測「最後一次 GPIO.output 是 LOW」這個格式（只測 call_count）。
# IT-5-A 確認 LED.cleanup() 直接呼叫時，green_pin 和 red_pin 都拿到 LOW。
# ---------------------------------------------------------------------------

class TestLedCleanupGpio:
    """LED.cleanup() 後，green_pin 和 red_pin 的 GPIO.output 最後是 LOW。"""

    def test_cleanup_drives_green_pin_low(self, led: LED, gpio_mock) -> None:
        """
        LED.cleanup() 必須呼叫 GPIO.output(green_pin, LOW)。

        green LED 亮起後若程式異常退出而未執行 cleanup，
        綠燈會永遠亮著（視覺上誤導授權狀態）。
        """
        led.cleanup()
        assert _pin_ends_low(gpio_mock, GREEN_LED_PIN), (
            f"cleanup() 後 GPIO pin {GREEN_LED_PIN}（green LED）應為 LOW"
        )

    def test_cleanup_drives_red_pin_low(self, led: LED, gpio_mock) -> None:
        """LED.cleanup() 必須呼叫 GPIO.output(red_pin, LOW)。"""
        led.cleanup()
        assert _pin_ends_low(gpio_mock, RED_LED_PIN), (
            f"cleanup() 後 GPIO pin {RED_LED_PIN}（red LED）應為 LOW"
        )

    def test_cleanup_calls_gpio_cleanup_for_both_pins(
        self, led: LED, gpio_mock
    ) -> None:
        """
        LED.cleanup() 必須呼叫 GPIO.cleanup([green_pin, red_pin]) 釋放資源。

        釋放 GPIO 資源讓其他程式（或下次啟動）可以重新配置同一個 pin，
        避免「RuntimeError: Pin already in use」。
        """
        led.cleanup()
        # GPIO.cleanup 可能以位置參數傳入 list
        cleanup_args = [
            c.args[0] if c.args else c.kwargs.get("channel_list")
            for c in gpio_mock.cleanup.call_args_list
        ]
        all_cleaned = [pin for arg in cleanup_args for pin in (arg if isinstance(arg, list) else [arg])]
        assert GREEN_LED_PIN in all_cleaned, f"GPIO.cleanup 未包含 green_pin ({GREEN_LED_PIN})"
        assert RED_LED_PIN in all_cleaned, f"GPIO.cleanup 未包含 red_pin ({RED_LED_PIN})"


# ---------------------------------------------------------------------------
# IT-5-B：Buzzer.cleanup() 的 GPIO 電位
#
# test_sensors.py 已有 test_cleanup_drives_pin_low_and_releases，
# 但使用 autouse reset_mock，與 ActuatorController 鏈的場景不同。
# IT-5-B 在相同的 gpio_mock 共享環境下驗證 Buzzer 的 cleanup 行為。
# ---------------------------------------------------------------------------

class TestBuzzerCleanupGpio:
    """Buzzer.cleanup() 後，buzzer_pin 的 GPIO.output 最後是 LOW。"""

    def test_cleanup_drives_buzzer_pin_low(
        self, buzzer: Buzzer, gpio_mock
    ) -> None:
        """
        Buzzer.cleanup() 必須呼叫 GPIO.output(BUZZER_PIN, LOW)。

        若 buzzer 被中斷時 pin 停在 HIGH，蜂鳴器永遠響。
        cleanup() 是確保硬體安全的最後防線。
        """
        buzzer.cleanup()
        assert _pin_ends_low(gpio_mock, BUZZER_PIN), (
            f"cleanup() 後 GPIO pin {BUZZER_PIN}（buzzer）應為 LOW"
        )

    def test_cleanup_calls_gpio_cleanup_for_buzzer_pin(
        self, buzzer: Buzzer, gpio_mock
    ) -> None:
        """Buzzer.cleanup() 必須呼叫 GPIO.cleanup([BUZZER_PIN]) 釋放資源。"""
        buzzer.cleanup()
        cleanup_args = [
            c.args[0] if c.args else c.kwargs.get("channel_list")
            for c in gpio_mock.cleanup.call_args_list
        ]
        all_cleaned = [pin for arg in cleanup_args for pin in (arg if isinstance(arg, list) else [arg])]
        assert BUZZER_PIN in all_cleaned, f"GPIO.cleanup 未包含 BUZZER_PIN ({BUZZER_PIN})"


# ---------------------------------------------------------------------------
# IT-5-C：ActuatorController.cleanup() 完整傳播鏈
#
# IT-5-C 是 IT-5 的核心：使用真實 LED + 真實 Buzzer，
# 驗證 actuator.cleanup() → led.cleanup() + buzzer.cleanup()
# 確實把所有 output pin 拉到 LOW。
#
# unit tests 的空缺：
#   test_actuator_controller.py TestCleanup 驗證 mock_led.cleanup() 被呼叫，
#   但 mock 的 cleanup() 不執行任何 GPIO 呼叫。
#   IT-5-C 讓真實的 LED 和 Buzzer 參與，確認 GPIO 電位在 cleanup 後確實為 LOW。
# ---------------------------------------------------------------------------

class TestActuatorCleanupChain:
    """actuator.cleanup() → 三個 output pin 全部被拉到 LOW。"""

    def test_cleanup_drives_green_led_pin_low(
        self, actuator: ActuatorController, gpio_mock
    ) -> None:
        """
        actuator.cleanup() → led.cleanup() → GPIO.output(green_pin, LOW)。

        這是 actuator → LED → GPIO 三層傳播的驗證。
        """
        actuator.cleanup()
        assert _pin_ends_low(gpio_mock, GREEN_LED_PIN), (
            "actuator.cleanup() 後 green LED pin 應為 LOW"
        )

    def test_cleanup_drives_red_led_pin_low(
        self, actuator: ActuatorController, gpio_mock
    ) -> None:
        """actuator.cleanup() → led.cleanup() → GPIO.output(red_pin, LOW)。"""
        actuator.cleanup()
        assert _pin_ends_low(gpio_mock, RED_LED_PIN), (
            "actuator.cleanup() 後 red LED pin 應為 LOW"
        )

    def test_cleanup_drives_buzzer_pin_low(
        self, actuator: ActuatorController, gpio_mock
    ) -> None:
        """actuator.cleanup() → buzzer.cleanup() → GPIO.output(BUZZER_PIN, LOW)。"""
        actuator.cleanup()
        assert _pin_ends_low(gpio_mock, BUZZER_PIN), (
            "actuator.cleanup() 後 buzzer pin 應為 LOW"
        )

    def test_cleanup_after_alert_does_not_raise(
        self, actuator: ActuatorController, gpio_mock
    ) -> None:
        """
        alert_unknown() 執行完後呼叫 cleanup() 不拋例外。

        這是 CAPSTONE demo 的真實場景：有人觸發警報後按 Ctrl-C，
        run() 的 finally 區塊呼叫 actuator.cleanup()。
        若 cleanup 在此時拋例外，GPIO 資源無法釋放。
        """
        with patch("time.sleep"):
            actuator.alert_unknown()

        gpio_mock.output.reset_mock()

        # 不應該拋任何例外
        actuator.cleanup()

        # cleanup 後 buzzer pin 仍必須是 LOW
        assert _pin_ends_low(gpio_mock, BUZZER_PIN)

    def test_cleanup_is_idempotent(
        self, actuator: ActuatorController, gpio_mock
    ) -> None:
        """
        連續呼叫 cleanup() 兩次不拋例外（冪等性）。

        edge case：若系統因為例外觸發兩次 cleanup（例如 finally 和 atexit），
        必須安全。mock GPIO 下 GPIO.cleanup() 是 no-op，所以測試通過；
        Jetson runner 上的 @pytest.mark.hardware 版本才會測真實 GPIO 的行為。
        """
        actuator.cleanup()
        actuator.cleanup()   # 第二次呼叫不應拋 RuntimeError / ValueError


# ---------------------------------------------------------------------------
# IT-5-D：cleanup 保護全部 output pin（整合性確認）
#
# 一個測試同時確認三個 pin 都被 cleanup，
# 讓 CI log 裡有一行清楚的「全部 pin 已保護」確認。
# ---------------------------------------------------------------------------

class TestAllPinsProtectedAfterCleanup:
    """actuator.cleanup() 後，所有 output pin 都在 LOW 狀態。"""

    def test_all_output_pins_are_low_after_cleanup(
        self, actuator: ActuatorController, gpio_mock
    ) -> None:
        """
        actuator.cleanup() 呼叫後：
          - GPIO pin 7  (green LED) 最後一次 output = LOW
          - GPIO pin 11 (red LED)   最後一次 output = LOW
          - GPIO pin 29 (buzzer)    最後一次 output = LOW

        這對應 CAPSTONE 的關鍵安全需求：
        Ctrl-C 後門鎖恢復上鎖狀態、LED 熄滅、buzzer 靜音。
        """
        actuator.cleanup()

        failed_pins = [
            pin for pin in _ALL_OUTPUT_PINS
            if not _pin_ends_low(gpio_mock, pin)
        ]
        assert not failed_pins, (
            f"cleanup() 後以下 GPIO pin 未回到 LOW：{failed_pins}\n"
            f"GPIO.output 呼叫記錄：{gpio_mock.output.call_args_list}"
        )
