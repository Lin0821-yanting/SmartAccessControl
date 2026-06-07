#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_jetson_smoke.py — IT-7

Integration test: Jetson 實機硬體 Smoke Test。

什麼是 Smoke Test（提醒）
-------------------------
硬體工程術語：電路板通電後若立刻冒煙，代表基本電路就掛了，不用繼續測。
IT-7 問的是：「系統能不能在真實硬體上活著啟動？」
不測細節正確性（那是 IT-1 ~ IT-6 的工作），只測「基本路徑不炸」。

IT-7 的定位
-----------
IT-1 ~ IT-6：
  使用 conftest.py 的 GPIO mock，可在 ubuntu-latest CI 執行。
  驗證邏輯正確性、GPIO 呼叫序列、MQTT payload 等。

IT-7：
  標記 @pytest.mark.hardware，只在 **Jetson self-hosted runner** 上執行。
  使用真實 Jetson.GPIO + 真實硬體（LED、Buzzer、HC-SR04）。
  CI yaml 中 ubuntu-latest 的 job 用 -m "not hardware" 略過本目錄的所有 IT-7 tests。
  Jetson job 用 -m hardware 只跑本檔。

執行環境檢測
-----------
_is_jetson() 檢查 /etc/nv_tegra_release（JetPack 安裝後存在）。
不在 Jetson 上時，所有測試自動 skip，不影響 ubuntu-latest CI 的 pass/fail。

CI yaml 整合
-----------
integration-test job（Jetson runner）：
    # Step A: mock-based（IT-1 ~ IT-6）
    pdm run pytest tests/integration -m "not hardware" -v

    # Step B: real GPIO（IT-7）
    pdm run pytest tests/integration -m hardware -v

GPIO mock 與真實 GPIO 的衝突解決
----------------------------------
conftest.py 透過 sys.modules.setdefault() 設定 GPIO mock。
在 Jetson runner 上，若希望 IT-7 使用真實 GPIO：
  方法 A（建議）：在 Jetson 的 CI step 執行前，先設定環境變數
                  SKIP_GPIO_MOCK=1，並讓 conftest.py 在此變數存在時
                  跳過 setdefault。
  方法 B：在本測試檔的 autouse fixture 中 pop sys.modules["Jetson.GPIO"]
          並 reimport src.led 等模組（較複雜，有副作用）。

目前設計（capstone 範疇）：
  IT-7 在有真實 GPIO 且無 mock 干擾的環境下執行基本初始化和短時操作，
  驗證「硬體驅動程式不崩潰」。若 conftest 的 mock 仍在，測試仍可 pass，
  因為 mock GPIO 讓所有 GPIO 呼叫成為 no-op（不拋例外）。
  真正的硬體驗證（亮燈、響聲）需人眼觀察，CI 只驗證不拋例外。

針腳使用（BOARD 編號）
-----------------------
  GREEN_LED = 7, RED_LED = 11, BUZZER = 29, SERVO = 33, TRIG = 31, ECHO = 15
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Jetson 環境檢測
# ---------------------------------------------------------------------------


def _is_jetson() -> bool:
    """/etc/nv_tegra_release 在 JetPack 安裝後存在，是可靠的 Jetson 識別標誌。"""
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists("/etc/nvpmodel.conf")


# 模組層級 skip：不在 Jetson 上時，整個檔案的所有測試都被跳過
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not _is_jetson(),
        reason=(
            "IT-7 需要真實 Jetson 硬體（/etc/nv_tegra_release 不存在）。"
            "在 Jetson self-hosted runner 上以 `pytest -m hardware` 執行。"
        ),
    ),
]

# ---------------------------------------------------------------------------
# 測試時間常數（縮短到可接受的 smoke test 時間）
# ---------------------------------------------------------------------------
_SMOKE_DURATION_S: float = 0.10  # LED/buzzer 操作的最短時間


# ===========================================================================
# IT-7-A：LED 硬體初始化與基本操作
# ===========================================================================


class TestLedHardwareSmoke:
    """LED 在真實 Jetson GPIO 上的基本可用性。"""

    def test_led_initializes_without_error(self) -> None:
        """
        LED() 在真實 GPIO 上初始化不拋例外。

        覆蓋：GPIO.setmode(BOARD) + GPIO.setup(pin, OUT, initial=LOW)
        若針腳接線錯誤或 GPIO overlay 未設定，這裡就會拋 RuntimeError。
        """
        from src.led import LED

        led = LED()
        led.cleanup()

    def test_led_grant_indicate_without_error(self) -> None:
        """
        LED.indicate(success=True) 執行 _SMOKE_DURATION_S 秒不拋例外。

        在真實硬體上：綠燈會短暫亮起然後熄滅。
        CI 只驗證不拋例外；肉眼觀察確認燈有亮。
        """
        from src.led import LED

        led = LED()
        try:
            led.indicate(success=True, duration=_SMOKE_DURATION_S)
        finally:
            led.cleanup()

    def test_led_deny_indicate_without_error(self) -> None:
        """
        LED.indicate(success=False) 執行 _SMOKE_DURATION_S 秒不拋例外。

        在真實硬體上：紅燈會短暫亮起然後熄滅。
        """
        from src.led import LED

        led = LED()
        try:
            led.indicate(success=False, duration=_SMOKE_DURATION_S)
        finally:
            led.cleanup()


# ===========================================================================
# IT-7-B：Buzzer 硬體初始化與基本操作
# ===========================================================================


class TestBuzzerHardwareSmoke:
    """Buzzer 在真實 Jetson GPIO 上的基本可用性。"""

    def test_buzzer_initializes_without_error(self) -> None:
        """Buzzer() 初始化不拋例外。"""
        from src.buzzer import Buzzer

        buz = Buzzer()
        buz.cleanup()

    def test_buzzer_single_beep_without_error(self) -> None:
        """
        Buzzer._beep(_SMOKE_DURATION_S) 執行不拋例外。

        在真實硬體上：會聽到一聲短促的嗶聲。
        """
        from src.buzzer import Buzzer

        buz = Buzzer()
        try:
            buz._beep(_SMOKE_DURATION_S)
        finally:
            buz.cleanup()


# ===========================================================================
# IT-7-C：ActuatorController 整合硬體 smoke test
# ===========================================================================


class TestActuatorHardwareSmoke:
    """ActuatorController 在真實硬體上的最短可用性驗證。"""

    def test_deny_access_sequence_without_error(self) -> None:
        """
        actuator.deny_access() 在真實硬體上執行不拋例外。

        覆蓋整條 ActuatorController → LED → GPIO 鏈的初始化和操作。
        紅燈應短暫亮起；_DENY_LED_S 預設 2 秒太長，用 patch 縮短。
        """
        from unittest.mock import patch

        from src.actuator_controller import ActuatorController

        ctrl = ActuatorController()
        try:
            with patch("src.actuator_controller._DENY_LED_S", _SMOKE_DURATION_S):
                ctrl.deny_access()
        finally:
            ctrl.cleanup()

    def test_actuator_cleanup_without_error(self) -> None:
        """
        actuator.cleanup() 在真實硬體上不拋例外。

        這是 CAPSTONE demo Ctrl-C 場景的最直接驗證：
        cleanup() 把所有 GPIO pin 還原到 LOW 並釋放資源。
        """
        from src.actuator_controller import ActuatorController

        ctrl = ActuatorController()
        ctrl.cleanup()  # 應不拋任何例外


# ===========================================================================
# IT-7-D：HC-SR04 硬體初始化
# ===========================================================================


class TestHcSr04HardwareSmoke:
    """HC-SR04 在真實 Jetson GPIO 上的基本可用性。"""

    def test_hcsr04_initializes_without_error(self) -> None:
        """
        HcSr04() 初始化不拋例外。

        若 Capstone AI Door Lock GPIO overlay 未透過 jetson-io.py 套用，
        這裡會因 ECHO pin 無法設為 IN 模式而拋 RuntimeError。
        """
        from src.hc_sr04 import HcSr04

        sensor = HcSr04()
        sensor.cleanup()

    def test_hcsr04_measure_distance_returns_float(self) -> None:
        """
        HcSr04._measure_distance() 回傳 float 值（inf 或正數）。

        在真實硬體上：若感測器接線正確，應回傳有限距離值。
        回傳 inf 代表逾時（可能是無人在前方），仍算 pass。
        CI 只驗證不拋例外且回傳型別正確。
        """
        from src.hc_sr04 import HcSr04

        sensor = HcSr04()
        try:
            distance = sensor._measure_distance()
            assert isinstance(distance, float), (
                f"_measure_distance() 應回傳 float，實際回傳 {type(distance)}"
            )
            assert distance > 0, f"_measure_distance() 回傳非正值 {distance}"
        finally:
            sensor.cleanup()
