#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_thread_safety.py — IT-6.

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

from src.actuator_controller import ActuatorController

# ---------------------------------------------------------------------------
# Fixture：帶有受控延遲的 ActuatorController
# ---------------------------------------------------------------------------


def _make_tracked_actuator(delay: float = 0.05) -> tuple[ActuatorController, list, threading.Event]:
    """
    建立一個真實 ActuatorController，其 LED mock 會：.

      1. 把 "START" 記錄到 call_log
      2. 設置 entered_event（通知測試執行緒「第一個方法已進入」）
      3. sleep delay 秒（模擬真實硬體佔用鎖的時間）
      4. 把 "END" 記錄到 call_log.

    回傳 (actuator, call_log, entered_event)。
    """
    call_log: list[str] = []
    entered_event = threading.Event()

    def tracked_led(success: bool, duration: float | None = None) -> None:
        call_log.append("START")
        entered_event.set()  # 通知：第一個方法已進入且持有 _lock
        time.sleep(delay)  # 模擬硬體佔用時間
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
# IT-6-A：同類方法並發時的 Drop-on-busy 行為
# ---------------------------------------------------------------------------


class TestSameMethodConcurrency:
    """測試 ActuatorController 的 Drop-on-busy 機制（非阻塞鎖）。."""

    def test_two_concurrent_deny_calls_first_wins_second_dropped(self) -> None:
        """兩個並發的 deny_access() 呼叫，預期只有第一個會執行，第二個被 silently dropped。."""
        actuator, call_log, entered = _make_tracked_actuator(delay=0.08)

        t1 = threading.Thread(target=actuator.deny_access, name="t1")
        t2 = threading.Thread(target=actuator.deny_access, name="t2")

        t1.start()
        entered.wait()          # t1 已進入並持有鎖
        t2.start()              # t2 嘗試取得鎖 → 應被 drop

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        # Drop-on-busy 行為：只有第一個呼叫執行
        assert call_log == ["START", "END"], (
            f"預期只有第一個 deny_access 執行，實際 call_log={call_log!r}"
        )

    def test_two_concurrent_alert_unknown_calls_first_wins_second_dropped(self) -> None:
        """兩個並發的 alert_unknown() 呼叫，第二個應被 dropped。."""
        actuator, call_log, entered = _make_tracked_actuator(delay=0.15)

        # 只 patch _multi_beep，不要 patch time.sleep（避免影響 tracked_led 的延遲）
        with patch.object(actuator, "_multi_beep"):
            t1 = threading.Thread(target=actuator.alert_unknown, name="t1")
            t2 = threading.Thread(target=actuator.alert_unknown, name="t2")

            t1.start()
            entered.wait()
            t2.start()

            t1.join(timeout=2.0)
            t2.join(timeout=1.0)

        assert call_log == ["START", "END"], (
            f"預期只有第一個 alert_unknown 執行，實際 call_log={call_log!r}"
        )


# ---------------------------------------------------------------------------
# IT-6-B：不同方法並發時的 Drop-on-busy 行為
# ---------------------------------------------------------------------------


class TestDifferentMethodConcurrency:
    """不同方法並發時，後發的呼叫應被 dropped（而非等待）。."""

    def test_deny_dropped_when_grant_is_ongoing(self) -> None:
        """grant_access() 執行中（持有鎖），並發的 deny_access() 應被 silently dropped。."""
        actuator, call_log, entered = _make_tracked_actuator(delay=0.08)

        t1 = threading.Thread(target=actuator.grant_access, name="grant")
        t2 = threading.Thread(target=actuator.deny_access, name="deny")

        t1.start()
        entered.wait()   # grant 已進入 indicate() 並持有鎖
        t2.start()       # deny 嘗試取得鎖 → 應被 drop

        t1.join(timeout=1.5)
        t2.join(timeout=1.0)

        # 只有 grant 的 LED 被執行，deny 被 drop
        assert call_log == ["START", "END"], (
            f"grant 執行中，deny 應被 dropped，實際 call_log={call_log!r}"
        )

    def test_alert_unknown_dropped_when_deny_is_ongoing(self) -> None:
        """deny_access() 執行中，後續的 alert_unknown() 應被 dropped。."""
        actuator, call_log, entered = _make_tracked_actuator(delay=0.15)

        with patch.object(actuator, "_multi_beep"):
            t1 = threading.Thread(target=actuator.deny_access, name="deny")
            t2 = threading.Thread(target=actuator.alert_unknown, name="unknown")

            t1.start()
            entered.wait()
            t2.start()

            t1.join(timeout=2.0)
            t2.join(timeout=1.0)

        assert call_log == ["START", "END"], (
            f"deny 執行中，alert_unknown 應被 dropped，實際 call_log={call_log!r}"
        )


# ---------------------------------------------------------------------------
# IT-6-C：多執行緒同時呼叫時的 Drop-on-busy 行為
# ---------------------------------------------------------------------------


class TestNoDroppedOperations:
    """多執行緒同時呼叫時，只有第一個能取得鎖，其餘全部被 dropped。."""

    def test_only_first_call_executes_when_many_concurrent(self) -> None:
        """五個執行緒同時呼叫 deny_access()，僅第一個成功，其餘被丟棄。."""
        call_log: list[str] = []
        log_lock = threading.Lock()

        def tracked_led(success: bool, duration: float | None = None) -> None:
            with log_lock:
                call_log.append("START")
            time.sleep(0.05)
            with log_lock:
                call_log.append("END")

        mock_led = MagicMock()
        mock_led.indicate.side_effect = tracked_led

        actuator = ActuatorController(
            led=mock_led, buzzer=MagicMock(), servo=MagicMock()
        )

        _n = 5
        threads = [threading.Thread(target=actuator.deny_access) for _ in range(_n)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        # Drop-on-busy 行為：只有 1 次執行
        assert call_log.count("START") == 1, (
            f"Drop-on-busy 模式下，預期只有 1 次 START，實際 {call_log.count('START')}"
        )
        assert call_log.count("END") == 1
        assert call_log == ["START", "END"]
