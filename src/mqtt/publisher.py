#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University 14210 AI實務專題
"""
publisher.py
------------
MQTT Publisher：將門禁事件、系統狀態、heartbeat 發送至 Kit#2。

Topics：
    lab/access/events    — 每次辨識決策結果
    lab/access/status    — 門鎖當前狀態
    lab/access/heartbeat — 系統健康資訊（fps、溫度、記憶體）

Usage:
    from src.mqtt.publisher import AccessPublisher
    pub = AccessPublisher(broker="localhost", port=1883)
    pub.publish_event(name="henry", similarity=0.97, granted=True)
"""

import json
import time
import threading
from datetime import datetime

import paho.mqtt.client as mqtt


class AccessPublisher:
    """
    MQTT 發布器，負責將門禁系統事件推送至 broker。

    Args:
        broker:   MQTT broker IP（預設 localhost）
        port:     MQTT broker port（預設 1883）
        topics:   topic 字典，key: events / status / heartbeat
    """

    DEFAULT_TOPICS = {
        "events": "lab/access/events",
        "status": "lab/access/status",
        "heartbeat": "lab/access/heartbeat",
    }

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topics: dict = None,
    ):
        self.broker = broker
        self.port = port
        self.topics = topics or self.DEFAULT_TOPICS

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._connected = False
        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────
    def _connect(self) -> None:
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            time.sleep(0.5)
        except Exception as e:
            print(f"[MQTT] 連線失敗：{e}（將以離線模式運行）")

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            self._connected = True
            print(f"[MQTT] 已連線：{self.broker}:{self.port}")
        else:
            print(f"[MQTT] 連線錯誤，reason_code={reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self._connected = False
        print(f"[MQTT] 已斷線，reason_code={reason_code}")

    def _publish(self, topic: str, payload: dict, qos: int = 1) -> None:
        """內部發布，離線時只印 log 不崩潰。"""
        msg = json.dumps(payload, ensure_ascii=False)
        if self._connected:
            self.client.publish(topic, msg, qos=qos)
        else:
            print(f"[MQTT][offline] {topic} → {msg}")

    # ── Public API ────────────────────────────────────────────────────────────
    def publish_event(
        self,
        name: str,
        similarity: float,
        liveness: float,
        granted: bool,
        reason: str = "",
    ) -> None:
        """
        發布單次辨識決策事件。

        Payload 格式：
            {
                "identity":   "henry",
                "similarity": 0.971,
                "liveness":   0.966,
                "granted":    true,
                "reason":     "similarity=0.971, liveness=0.966",
                "timestamp":  "2026-06-05T15:42:06"
            }
        """
        payload = {
            "identity": name,
            "similarity": round(similarity, 4),
            "liveness": round(liveness, 4),
            "granted": granted,
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._publish(self.topics["events"], payload)

    def publish_status(self, door_state: str, last_person: str = "") -> None:
        """
        發布門鎖狀態。

        Args:
            door_state:  "locked" 或 "unlocked"
            last_person: 最後授權的人員名稱
        """
        payload = {
            "door_state": door_state,
            "last_person": last_person,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._publish(self.topics["status"], payload)

    def publish_heartbeat(
        self,
        fps: float,
        distance_cm: float = -1,
    ) -> None:
        """
        發布系統健康資訊。

        Args:
            fps:         當前推論 FPS
            distance_cm: HC-SR04 距離（-1 表示未啟用）
        """
        payload = {
            "fps": round(fps, 1),
            "distance_cm": round(distance_cm, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._publish(self.topics["heartbeat"], payload)

    def start_heartbeat(self, fps_getter, interval: float = 5.0) -> None:
        """
        背景執行緒定期發送 heartbeat。

        Args:
            fps_getter: callable，回傳當前 fps（float）
            interval:   發送間隔秒數（預設 5 秒）
        """

        def _loop():
            while True:
                self.publish_heartbeat(fps=fps_getter())
                time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        print(f"[MQTT] Heartbeat 啟動，每 {interval} 秒發送一次")

    def disconnect(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        print("[MQTT] 已中斷連線")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pub = AccessPublisher()
    time.sleep(1)

    print("\n[TEST] 發送 event（授權）")
    pub.publish_event("henry", 0.971, 0.966, True, "similarity=0.971, liveness=0.966")

    print("[TEST] 發送 event（拒絕）")
    pub.publish_event("unknown", 0.612, 0.342, False, "spoof detected")

    print("[TEST] 發送 status")
    pub.publish_status("unlocked", "henry")

    print("[TEST] 發送 heartbeat")
    pub.publish_heartbeat(fps=38.5, distance_cm=45.2)

    time.sleep(1)
    pub.disconnect()
