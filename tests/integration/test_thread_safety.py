#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_thread_safety.py — IT-6

Integration test: ActuatorController._lock (RLock) 執行緒安全。

現有測試的空缺
--------------
  test_actuator_controller.py  : 每個方法單獨執行，無並發場景
  IT-1 ~ IT-5                  : 每次只有一個執行緒操作 actuator

ActuatorController docstring 明確說明：
  "Thread safety: each method acquires a reentrant lock so overlapping
   decisions (e.g. rapid-fire deny during an ongoing grant) do not race."

IT-6 填補的空缺
---------------
Orchestrator._act() 把每個決策分派到獨立的 daemon thread：

    threading.Thread(target=self._actuator.deny_access, daemon=True).start()
    threading.Thread(target=self._actuator.alert_unknown, daemon=True).start()

若多幀決策快速連發（例如連續兩幀都偵測到 DENY），actuator 會收到來自兩個
執行緒的並發呼叫。IT-6 驗證 _lock 確實讓這些呼叫序列化，不會出現：
  - LED HIGH 出現兩次才出現第一次 LOW（電位混亂）
  - buzzer 的 HIGH/LOW 序列被另一個方法的呼叫打斷

測試策略
--------
在 mock LED 的 indicate() 中插入 threading.Event + time.sleep，
製造「第一個方法正在持有 _lock 執行中」的場景，然後啟動第二個執行緒。

若 _lock 正確運作，第二個執行緒必須等第一個方法完全結束後才能開始執行。
這反映在 call_log 上的模式：

    正確（序列化）：["START", "END", "START", "END"]
    錯誤（交錯）：  ["START", "START", "END", "END"]

Mock 邊界
---------
  真實執行：ActuatorController、threading.RLock、所有 public 方法
  Mock：     LED.indicate（含受控延遲）、Buzzer._beep、Servo.unlock_then_relock

CI 相容性：ubuntu-latest 可執行，不依賴 GPIO。
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.actuator_controller import ActuatorController


# ---------------------------------------------------------------------------
# Fixture：帶有受控延遲的 ActuatorController
# ---------------------------------------------------------------------------

def _make_tracked_actuator(delay: float = 0.05) -> tuple[ActuatorController, list, threading.Event]:
    """
    建立一個真實 ActuatorController，其 LED mock 會：
      1. 把 "START" 記錄到 call_log
      2. 設置 entered_event（通知測試執行緒「第一個方法已進入」）
      3. sleep delay 秒（模擬真實硬體佔用鎖的時間）
      4. 把 "END" 記錄到 call_log

    回傳 (actuator, call_log, entered_event)。
    """
    call_log: list[str] = []
    entered_event = threading.Event()

    def tracked_led(success: bool, duration: float | None = None) -> None:  # noqa: ANN001
        call_log.append("START")
        entered_event.set()      # 通知：第一個方法已進入且持有 _lock
        time.sleep(delay)        # 模擬硬體佔用時間
        call_log.append("END")

    mock_led = MagicMock()
    mock_led.indicate.side_effect = tracked_led

    actuator = ActuatorController(
        led=mock_led,
        buzzer=MagicMock(),
        servo=MagicMock(),
    )
    return actuator, call_log, entered_event


# ---------------------------------------------------------------------------
# IT-6-A：同類方法並發序列化
#
# 最常見的場景：同一人連續兩幀都被判為 DENY，
# orchestrator 連發兩個 daemon thread 各呼叫一次 deny_access()。
# ---------------------------------------------------------------------------

class TestSameMethodConcurrency:
    """兩個並發的相同 actuator 方法必須序列化執行。"""

    def test_two_concurrent_deny_calls_serialize(self) -> None:
        """
        t1 持有 _lock 執行 deny_access() 期間，t2 被阻擋。
        t1 釋放 _lock 後，t2 才能開始執行。

        預期 call_log：["START", "END", "START", "END"]（序列化）
        拒絕 call_log：["START", "START", "END", "END"]（交錯 → RLock 失效）
        """
        actuator, call_log, entered = _make_tracked_actuator(delay=0.05)

        t1 = threading.Thread(target=actuator.deny_access, name="t1")
        t2 = threading.Thread(target=actuator.deny_access, name="t2")

        t1.start()
        entered.wait()   # t1 已持有 _lock 並進入 indicate()
        t2.start()       # t2 嘗試取得 _lock，應被阻擋

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        assert call_log == ["START", "END", "START", "END"], (
            f"deny_access() 並發呼叫未序列化，call_log={call_log!r}"
        )

    def test_two_concurrent_alert_unknown_calls_serialize(self) -> None:
        """
        連續兩次 alert_unknown() 的 LED 呼叫必須序列化。
        """
        actuator, call_log, entered = _make_tracked_actuator(delay=0.05)

        with patch("time.sleep"):   # 抑制 _multi_beep 的 inter-beep sleep
            t1 = threading.Thread(target=actuator.alert_unknown, name="t1")
            t2 = threading.Thread(target=actuator.alert_unknown, name="t2")

            t1.start()
            entered.wait()
            t2.start()

            t1.join(timeout=1.0)
            t2.join(timeout=1.0)

        assert call_log == ["START", "END", "START", "END"], (
            f"alert_unknown() 並發呼叫未序列化，call_log={call_log!r}"
        )


# ---------------------------------------------------------------------------
# IT-6-B：不同方法並發序列化
#
# 更真實的場景：GRANT 的 daemon thread 尚未完成，下一幀卻產生 DENY，
# orchestrator 再發一個 daemon thread 呼叫 deny_access()。
# 若沒有 _lock，DENY 的紅燈會在 GRANT 的綠燈還亮著時就啟動。
# ---------------------------------------------------------------------------

class TestDifferentMethodConcurrency:
    """GRANT 執行中，並發的 DENY 必須等 GRANT 完成後才開始。"""

    def test_deny_waits_for_ongoing_grant(self) -> None:
        """
        t1 執行 grant_access()（持有 _lock）時，
        t2 執行 deny_access() 必須被阻擋。

        驗證：call_log 中不會出現 ["START", "START", ...] 的模式。
        """
        actuator, call_log, entered = _make_tracked_actuator(delay=0.05)

        t1 = threading.Thread(target=actuator.grant_access, name="grant")
        t2 = threading.Thread(target=actuator.deny_access, name="deny")

        t1.start()
        entered.wait()   # grant 已持有 _lock
        t2.start()       # deny 嘗試取得 _lock，應被阻擋

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        # grant 的 LED START 後，deny 的 LED START 必須在 grant 的 END 之後
        assert call_log == ["START", "END", "START", "END"], (
            f"grant_access() 與 deny_access() 並發執行未序列化，call_log={call_log!r}"
        )

    def test_unknown_alert_waits_for_ongoing_deny(self) -> None:
        """
        DENY 執行中，後續 UNKNOWN 必須等待，不能打斷 DENY 的紅燈序列。
        """
        actuator, call_log, entered = _make_tracked_actuator(delay=0.05)

        with patch("time.sleep"):
            t1 = threading.Thread(target=actuator.deny_access, name="deny")
            t2 = threading.Thread(target=actuator.alert_unknown, name="unknown")

            t1.start()
            entered.wait()
            t2.start()

            t1.join(timeout=1.0)
            t2.join(timeout=1.0)

        assert call_log == ["START", "END", "START", "END"], (
            f"deny_access() 與 alert_unknown() 並發執行未序列化，call_log={call_log!r}"
        )


# ---------------------------------------------------------------------------
# IT-6-C：完成所有並發呼叫後無遺漏操作
#
# 序列化保證「第一個完整執行、第二個完整執行」，
# 不會出現「第一個執行到一半被跳過」的情況。
# ---------------------------------------------------------------------------

class TestNoDroppedOperations:
    """並發呼叫都必須完整執行，不能因鎖競爭而被略過。"""

    def test_all_concurrent_calls_complete(self) -> None:
        """
        5 個執行緒同時呼叫 deny_access()，
        所有 5 次 LED.indicate() 都必須完整執行（不能有呼叫被吞掉）。
        """
        call_log: list[str] = []
        lock = threading.Lock()

        def tracked_led(success: bool, duration: float | None = None) -> None:  # noqa: ANN001
            with lock:
                call_log.append("START")
            time.sleep(0.02)
            with lock:
                call_log.append("END")

        mock_led = MagicMock()
        mock_led.indicate.side_effect = tracked_led

        actuator = ActuatorController(
            led=mock_led, buzzer=MagicMock(), servo=MagicMock()
        )

        _N = 5
        threads = [
            threading.Thread(target=actuator.deny_access)
            for _ in range(_N)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert call_log.count("START") == _N, (
            f"預期 {_N} 次 START，實際 {call_log.count('START')}"
        )
        assert call_log.count("END") == _N, (
            f"預期 {_N} 次 END，實際 {call_log.count('END')}"
        )
        # 序列化：START 和 END 必須交替出現，不能有連續兩個 START
        for i in range(0, len(call_log) - 1, 2):
            assert call_log[i] == "START" and call_log[i + 1] == "END", (
                f"位置 {i} 出現非 START/END 交替的模式：{call_log}"
            )
