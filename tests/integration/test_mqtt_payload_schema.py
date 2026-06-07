#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/integration/test_mqtt_payload_schema.py — IT-4

Integration test: MQTT payload 欄位正確性（AI pipeline → broker 的完整鏈）。

現有測試的覆蓋邊界
------------------
  test_mqtt_publisher.py
      直接呼叫 ``publisher.publish_event(decision="GRANT", identity="alice", ...)``
      → 驗證 JSON 編碼正確、欄位齊全、數值有四捨五入。
      ✗ 未測「AI pipeline 輸出的值有沒有正確流入 publish_event」。

  test_orchestrator.py
      驗證 ``mock_publisher.publish_event.call_args.kwargs["decision"] == "GRANT"``
      → 以 MagicMock publisher 確認呼叫發生。
      ✗ 未測最終送到 broker 的 JSON 字串。

IT-4 填補的空缺
---------------
IT-4 使用**真實 MqttPublisher**（配合 mock paho client），讓資料流跑完：

    recog.name / recog.similarity / recog.authorized
    liveness.is_live / liveness.score
    face.bbox (numpy array)
    DecisionEngine.evaluate()
        └→ Orchestrator._publish_event_from()
            └→ MqttPublisher.publish_event()
                └→ mock_paho_client.publish(topic, JSON_STRING, qos=...)

最後從 mock_paho_client.publish.call_args_list 取出 JSON 字串並 parse，
驗證每個欄位的值確實來自正確的 AI mock 屬性（而非隨意命名的預設值）。

典型的 bug 場景（這些 bug 讓 IT-4 抓到，test_mqtt_publisher.py 看不到）：
  • _publish_event_from() 把 liveness.score 傳給 similarity 而非 spoof_score
  • face.bbox 沒有 astype(int).tolist()，broker 收到 float array 的字串
  • recog.identity 被誤用為 identity 而非 recog.name
  • DENY 時 face_in_db 被誤設為 False

Mock 邊界
---------
  真實執行：MqttPublisher、Orchestrator._tick() / _publish_event_from()
             DecisionEngine
  Mock：     paho.mqtt.Client（攔截 publish() 呼叫，解析 JSON 字串）
             ActuatorController（讓 thread 即時完成）
             FaceDetector、FaceRecognizer、AntiSpoof、HCSR04
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.decision_engine import DecisionEngine
from src.mqtt_publisher import (
    QOS_STATUS,
    TOPIC_EVENTS,
    TOPIC_STATUS,
    MqttPublisher,
)
from src.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# 假資料建構函式（與前幾個 IT 保持相同命名）
# ---------------------------------------------------------------------------

_FACE_BBOX = np.array([10, 20, 110, 120], dtype=np.float32)
_FACE_BBOX_AS_INT_LIST = [10, 20, 110, 120]  # 預期 broker 看到的值


def _blank() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_face() -> MagicMock:
    """Bbox 使用固定值，方便 IT-4 斷言 broker 收到的 bbox 欄位。"""
    face = MagicMock()
    face.bbox = _FACE_BBOX.copy()
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
def mock_paho_client() -> MagicMock:
    """
    Mock paho Client。每次 publish 回傳 rc=0（MQTT_ERR_SUCCESS）。

    所有 publish() 呼叫記錄在 mock_paho_client.publish.call_args_list，
    call_args_list[i].args = (topic, json_string)
    call_args_list[i].kwargs = {"qos": N}
    """
    client = MagicMock()
    result = MagicMock()
    result.rc = 0
    client.publish.return_value = result
    return client


@pytest.fixture()
def publisher(mock_paho_client: MagicMock) -> MqttPublisher:
    """
    真實 MqttPublisher，使用 mock paho client。
    ``_connected = True`` 讓 publish() 方法真正呼叫 client.publish()。
    """
    pub = MqttPublisher(client_factory=lambda: mock_paho_client)
    pub._connected = True
    return pub


@pytest.fixture()
def engine() -> DecisionEngine:
    return DecisionEngine(similarity_threshold=0.85, required_frames=3)


@pytest.fixture()
def ai_mocks() -> dict:
    return {
        "detector": MagicMock(),
        "recognizer": MagicMock(),
        "antispoof": MagicMock(),
    }


@pytest.fixture()
def orc(engine: DecisionEngine, publisher: MqttPublisher, ai_mocks: dict) -> Orchestrator:
    """Orchestrator：真實 engine + 真實 publisher（mock paho），mock actuator / AI。"""
    return Orchestrator(
        detector=ai_mocks["detector"],
        recognizer=ai_mocks["recognizer"],
        antispoof=ai_mocks["antispoof"],
        engine=engine,
        actuator=MagicMock(),
        publisher=publisher,
        sensor=MagicMock(),
        display=False,
    )


# ---------------------------------------------------------------------------
# 輔助：驅動 _tick() 與解析 broker 的 JSON
# ---------------------------------------------------------------------------


def _tick_matching(
    orc: Orchestrator,
    ai_mocks: dict,
    recog: MagicMock | None = None,
    liveness: MagicMock | None = None,
) -> None:
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = recog or _make_recog()
    ai_mocks["antispoof"].predict.return_value = liveness or _make_liveness()
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_deny(orc: Orchestrator, ai_mocks: dict) -> None:
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog(similarity=0.70, authorized=True)
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=True)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_unknown(orc: Orchestrator, ai_mocks: dict) -> None:
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog(
        name="stranger", similarity=0.50, authorized=False
    )
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=True)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_spoof(orc: Orchestrator, ai_mocks: dict) -> None:
    ai_mocks["detector"].detect.return_value = [_make_face()]
    ai_mocks["recognizer"].match.return_value = _make_recog()
    ai_mocks["antispoof"].predict.return_value = _make_liveness(is_live=False, score=0.15)
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _tick_no_face(orc: Orchestrator, ai_mocks: dict) -> None:
    ai_mocks["detector"].detect.return_value = []
    with patch.object(orc._sensor, "_measure_distance", return_value=30.0):
        orc._tick(_blank())


def _payloads_on_topic(mock_paho_client: MagicMock, topic: str) -> list[dict]:
    """回傳所有發送到 topic 的 JSON payload（依序）。"""
    return [
        json.loads(c.args[1])
        for c in mock_paho_client.publish.call_args_list
        if c.args and c.args[0] == topic
    ]


def _last_event(mock_paho_client: MagicMock) -> dict:
    """取出最後一個 lab/access/events payload。"""
    payloads = _payloads_on_topic(mock_paho_client, TOPIC_EVENTS)
    assert payloads, "沒有任何 TOPIC_EVENTS 發送記錄"
    return payloads[-1]


def _last_status(mock_paho_client: MagicMock) -> dict:
    """取出最後一個 lab/access/status payload。"""
    payloads = _payloads_on_topic(mock_paho_client, TOPIC_STATUS)
    assert payloads, "沒有任何 TOPIC_STATUS 發送記錄"
    return payloads[-1]


# ---------------------------------------------------------------------------
# IT-4-A：GRANT event payload 欄位準確性
#
# 驗證 AI mock 的屬性值有沒有被正確對應到 broker 收到的 JSON 欄位。
# 每個測試驗證一個映射關係，失敗訊息明確指向哪個欄位出問題。
#
# 現有測試的空缺：
#   test_mqtt_publisher.py 直接傳 decision="GRANT"、identity="alice" 給 publish_event()，
#   不測「Orchestrator 有沒有把 recog.name 放進 identity 欄位」。
# ---------------------------------------------------------------------------


class TestGrantEventPayload:
    """GRANT 決策的 lab/access/events JSON 欄位準確對應 AI mock 屬性。"""

    _RECOG_NAME = "bob"
    _RECOG_SIMILARITY = 0.9312
    _LIVENESS_SCORE = 0.875

    @pytest.fixture(autouse=True)
    def _drive_grant(self, orc: Orchestrator, ai_mocks: dict) -> None:
        """送 3 幀讓 engine 達到 GRANT，並等 actuator thread 完成。"""
        recog = _make_recog(
            name=self._RECOG_NAME,
            similarity=self._RECOG_SIMILARITY,
            authorized=True,
        )
        liveness = _make_liveness(is_live=True, score=self._LIVENESS_SCORE)
        for _ in range(3):
            _tick_matching(orc, ai_mocks, recog=recog, liveness=liveness)
        time.sleep(0.05)

    def test_decision_field_is_grant(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["decision"] == "GRANT"

    def test_identity_maps_from_recog_name(self, mock_paho_client: MagicMock) -> None:
        """
        Identity 欄位來自 recog.name（而非 recog.identity 或其他屬性）。
        這是 IT-4 最核心的驗證：確認 _publish_event_from(identity=recog.name, ...) 正確。
        """
        assert _last_event(mock_paho_client)["identity"] == self._RECOG_NAME

    def test_similarity_maps_from_recog_similarity(self, mock_paho_client: MagicMock) -> None:
        """Similarity 欄位來自 recog.similarity，四捨五入到 4 位小數。"""
        expected = round(self._RECOG_SIMILARITY, 4)
        assert _last_event(mock_paho_client)["similarity"] == pytest.approx(expected)

    def test_spoof_score_maps_from_liveness_score(self, mock_paho_client: MagicMock) -> None:
        """
        spoof_score 欄位來自 liveness.score（不是 liveness.is_live）。
        若 _publish_event_from() 把 similarity 和 spoof_score 傳錯，這個測試攔截。
        """
        expected = round(self._LIVENESS_SCORE, 4)
        assert _last_event(mock_paho_client)["spoof_score"] == pytest.approx(expected)

    def test_is_live_maps_from_liveness_is_live(self, mock_paho_client: MagicMock) -> None:
        """is_live 欄位來自 liveness.is_live（bool）。"""
        assert _last_event(mock_paho_client)["is_live"] is True

    def test_face_in_db_maps_from_recog_authorized(self, mock_paho_client: MagicMock) -> None:
        """face_in_db 欄位來自 recog.authorized（不是 recog.in_db 或其他）。"""
        assert _last_event(mock_paho_client)["face_in_db"] is True

    def test_bbox_is_int_list_from_face_bbox(self, mock_paho_client: MagicMock) -> None:
        """
        Bbox 欄位是 [x1, y1, x2, y2] 整數 list（不是 float array 字串）。

        _tick() 中執行：bbox = face.bbox.astype(int).tolist()
        若這行被移除或型別轉換錯誤，broker 收到的就不是整數 list。
        """
        bbox = _last_event(mock_paho_client)["bbox"]
        assert bbox == _FACE_BBOX_AS_INT_LIST
        assert all(isinstance(v, int) for v in bbox)

    def test_consecutive_frames_is_zero_after_grant(self, mock_paho_client: MagicMock) -> None:
        """
        GRANT event 的 consecutive_frames 欄位是 0。

        DecisionEngine 在回傳 GRANT 前呼叫 _reset()，
        所以 _publish_event_from() 讀到的 engine.consecutive_frames 已是 0。
        """
        assert _last_event(mock_paho_client)["consecutive_frames"] == 0


# ---------------------------------------------------------------------------
# IT-4-B：DENY event payload
#
# 驗證 DENY 時 is_live=True、face_in_db=True（臉在 DB 但相似度不夠）。
# 這是一個容易弄錯的語意：DENY ≠ 臉不在 DB，只是相似度不足。
# ---------------------------------------------------------------------------


class TestDenyEventPayload:
    """DENY 決策的 lab/access/events JSON 欄位（face_in_db=True、is_live=True）。"""

    @pytest.fixture(autouse=True)
    def _drive_deny(self, orc: Orchestrator, ai_mocks: dict) -> None:
        _tick_deny(orc, ai_mocks)

    def test_decision_is_deny(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["decision"] == "DENY"

    def test_face_in_db_is_true_on_deny(self, mock_paho_client: MagicMock) -> None:
        """
        DENY 時 face_in_db 必須是 True（臉在 DB，但相似度不夠）。
        若誤設為 False，Kit #2 的 dashboard 會誤以為是陌生人。
        """
        assert _last_event(mock_paho_client)["face_in_db"] is True

    def test_is_live_is_true_on_deny(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["is_live"] is True


# ---------------------------------------------------------------------------
# IT-4-C：UNKNOWN event payload
#
# 驗證 UNKNOWN 時 face_in_db=False、is_live=True。
# ---------------------------------------------------------------------------


class TestUnknownEventPayload:
    """UNKNOWN 決策的 lab/access/events JSON 欄位（face_in_db=False）。"""

    @pytest.fixture(autouse=True)
    def _drive_unknown(self, orc: Orchestrator, ai_mocks: dict) -> None:
        _tick_unknown(orc, ai_mocks)

    def test_decision_is_unknown(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["decision"] == "UNKNOWN"

    def test_face_in_db_is_false_on_unknown(self, mock_paho_client: MagicMock) -> None:
        """
        UNKNOWN 時 face_in_db 必須是 False（臉不在 DB）。
        這對應 recog.authorized=False，由 _tick() 轉成 face_in_db=False。
        """
        assert _last_event(mock_paho_client)["face_in_db"] is False

    def test_identity_is_stranger_name(self, mock_paho_client: MagicMock) -> None:
        """即使 UNKNOWN，identity 欄位仍帶著 recog.name（方便 log 追蹤）。"""
        assert _last_event(mock_paho_client)["identity"] == "stranger"


# ---------------------------------------------------------------------------
# IT-4-D：SPOOF event payload
#
# 驗證 SPOOF 時 is_live=False、spoof_score 低值正確傳遞。
# ---------------------------------------------------------------------------


class TestSpoofEventPayload:
    """SPOOF 決策的 lab/access/events JSON 欄位（is_live=False）。"""

    _SPOOF_SCORE = 0.15

    @pytest.fixture(autouse=True)
    def _drive_spoof(self, orc: Orchestrator, ai_mocks: dict) -> None:
        _tick_spoof(orc, ai_mocks)

    def test_decision_is_spoof(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["decision"] == "SPOOF"

    def test_is_live_is_false_on_spoof(self, mock_paho_client: MagicMock) -> None:
        """
        SPOOF 時 is_live 必須是 False。
        這由 _tick() 的 anti_spoof_pass = liveness.is_live 取得。
        """
        assert _last_event(mock_paho_client)["is_live"] is False

    def test_spoof_score_value(self, mock_paho_client: MagicMock) -> None:
        """spoof_score 帶著真實的低分（0.15），方便 Kit #2 知道置信度。"""
        assert _last_event(mock_paho_client)["spoof_score"] == pytest.approx(
            round(self._SPOOF_SCORE, 4)
        )


# ---------------------------------------------------------------------------
# IT-4-E：無臉幀的 event payload
#
# 無人臉時 _tick() 呼叫 engine.ignore()，仍然發送一個 IGNORE 事件，
# bbox 為 None、identity 為 "unknown"。
# ---------------------------------------------------------------------------


class TestNoFaceEventPayload:
    """gate open 但偵測不到臉 → publish IGNORE event，bbox=None。"""

    @pytest.fixture(autouse=True)
    def _drive_no_face(self, orc: Orchestrator, ai_mocks: dict) -> None:
        _tick_no_face(orc, ai_mocks)

    def test_decision_is_ignore_when_no_face(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["decision"] == "IGNORE"

    def test_bbox_is_none_when_no_face(self, mock_paho_client: MagicMock) -> None:
        """無臉時 bbox 必須是 JSON null（Python None），不能是空 list 或省略。"""
        assert _last_event(mock_paho_client)["bbox"] is None

    def test_identity_is_unknown_when_no_face(self, mock_paho_client: MagicMock) -> None:
        assert _last_event(mock_paho_client)["identity"] == "unknown"


# ---------------------------------------------------------------------------
# IT-4-F：lab/access/status payload 與 QoS
#
# publish_status 在 GRANT 後被呼叫（door_state="unlocked", last_person=recog.name）。
# QoS 必須是 1（門鎖狀態訊息，不允許 drop）。
# ---------------------------------------------------------------------------


class TestStatusPayload:
    """lab/access/status 的 payload 欄位與 QoS。"""

    _PERSON_NAME = "carol"

    @pytest.fixture(autouse=True)
    def _drive_grant(self, orc: Orchestrator, ai_mocks: dict) -> None:
        recog = _make_recog(name=self._PERSON_NAME, similarity=0.92)
        liveness = _make_liveness(is_live=True)
        for _ in range(3):
            _tick_matching(orc, ai_mocks, recog=recog, liveness=liveness)
        time.sleep(0.05)

    def test_door_state_is_unlocked_after_grant(self, mock_paho_client: MagicMock) -> None:
        assert _last_status(mock_paho_client)["door_state"] == "unlocked"

    def test_last_person_maps_from_recog_name(self, mock_paho_client: MagicMock) -> None:
        """
        last_person 欄位必須是 recog.name，而非硬編碼的 "unknown" 或其他值。
        Orchestrator._set_door_state(state, identity) 中 identity 來自 recog.name。
        """
        assert _last_status(mock_paho_client)["last_person"] == self._PERSON_NAME

    def test_status_published_with_qos_one(self, mock_paho_client: MagicMock) -> None:
        """
        lab/access/status 的 QoS 必須是 1。
        門鎖狀態屬於高重要性訊息（proposal §4.7：status QoS=1）。
        """
        status_calls = [
            c
            for c in mock_paho_client.publish.call_args_list
            if c.args and c.args[0] == TOPIC_STATUS
        ]
        assert status_calls, "沒有任何 TOPIC_STATUS 發送記錄"
        last = status_calls[-1]
        # paho.publish(topic, body, qos=N) — qos 可以是位置或 keyword 參數
        actual_qos = last.kwargs.get("qos") if last.kwargs.get("qos") is not None else last.args[2]
        assert actual_qos == QOS_STATUS


# ---------------------------------------------------------------------------
# IT-4-G：timestamp 欄位格式
#
# 所有三個 topic 的 payload 都含有 timestamp 欄位。
# IT-4-G 驗證 events topic 的 timestamp 是合法的 UTC ISO 8601 字串，
# 可以被 datetime.fromisoformat() 解析，且帶有時區資訊。
#
# _now_iso() 使用 timezone.utc，產生如 "2026-06-07T12:34:56.789+00:00" 的格式。
# ---------------------------------------------------------------------------


class TestTimestampFormat:
    """publish_event 的 timestamp 欄位是有效的 UTC ISO 8601 字串。"""

    def test_event_timestamp_is_parseable_utc_iso8601(
        self, orc: Orchestrator, ai_mocks: dict, mock_paho_client: MagicMock
    ) -> None:
        """
        Timestamp 欄位必須：
          1. 能被 datetime.fromisoformat() 解析
          2. 含有 UTC 時區資訊（tzinfo 非 None）
        """
        _tick_deny(orc, ai_mocks)
        ts_str = _last_event(mock_paho_client)["timestamp"]

        dt = datetime.fromisoformat(ts_str)
        assert dt.tzinfo is not None, f"timestamp '{ts_str}' 缺少時區資訊（應為 UTC）"
        # UTC 時區的 utcoffset() 應為 timedelta(0)
        from datetime import timedelta

        assert dt.utcoffset() == timedelta(0), f"timestamp '{ts_str}' 的時區不是 UTC"
