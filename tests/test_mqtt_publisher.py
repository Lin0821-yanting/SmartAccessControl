#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""tests/test_mqtt_publisher.py — unit tests for MqttPublisher.

All tests use unittest.mock to stub paho-mqtt so no real broker is needed.
Run with:
    pytest tests/test_mqtt_publisher.py -v
"""

from __future__ import annotations

import json
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from src.mqtt_publisher import (
    QOS_STATUS,
    TOPIC_EVENTS,
    TOPIC_HEARTBEAT,
    TOPIC_STATUS,
    MqttPublisher,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a mock paho Client with publish() returning a success MagicMock."""
    client = MagicMock()
    result = MagicMock()
    result.rc = 0  # MQTT_ERR_SUCCESS
    client.publish.return_value = result
    return client


@pytest.fixture()
def publisher(mock_client: MagicMock) -> MqttPublisher:
    """MqttPublisher wired with the mock client (not connected)."""
    return MqttPublisher(client_factory=lambda: mock_client)


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------


class TestConnectionState:
    def test_initially_disconnected(self, publisher: MqttPublisher) -> None:
        assert publisher.connected is False

    def test_connect_calls_paho_connect_and_loop_start(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        # Force connected flag so connect() doesn't time-out waiting
        def fake_connect(*args, **kwargs):
            publisher._connected = True

        mock_client.connect.side_effect = fake_connect
        publisher.connect()

        mock_client.connect.assert_called_once()
        mock_client.loop_start.assert_called_once()
        assert publisher.connected is True

    def test_disconnect_stops_loop_and_clears_flag(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True  # simulate connected state
        publisher.disconnect()

        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
        assert publisher.connected is False

    def test_reconnect_delay_set_called_on_init(self, mock_client: MagicMock) -> None:
        MqttPublisher(client_factory=lambda: mock_client)
        mock_client.reconnect_delay_set.assert_called_once()


# ---------------------------------------------------------------------------
# publish() low-level method
# ---------------------------------------------------------------------------


class TestPublishLowLevel:
    def test_publish_when_disconnected_returns_false(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        result = publisher.publish(TOPIC_EVENTS, {"x": 1})
        assert result is False
        mock_client.publish.assert_not_called()

    def test_publish_json_encodes_dict(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        payload = {"decision": "GRANT", "identity": "alice"}

        publisher.publish(TOPIC_EVENTS, payload)

        args, _ = mock_client.publish.call_args
        topic, body = args[0], args[1]
        assert topic == TOPIC_EVENTS
        assert json.loads(body) == payload

    def test_publish_string_payload_not_double_encoded(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        raw = '{"already": "json"}'
        publisher.publish(TOPIC_EVENTS, raw)
        args, _ = mock_client.publish.call_args
        assert args[1] == raw  # not '\'{"already": "json"}\''

    def test_publish_returns_false_on_paho_error(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        error_result = MagicMock()
        error_result.rc = 4  # MQTT_ERR_NO_CONN
        mock_client.publish.return_value = error_result

        result = publisher.publish(TOPIC_EVENTS, {"x": 1})
        assert result is False


# ---------------------------------------------------------------------------
# publish_event()
# ---------------------------------------------------------------------------


class TestPublishEvent:
    _BASE_KWARGS: ClassVar[dict] = dict(
        decision="GRANT",
        identity="alice",
        similarity=0.921,
        spoof_score=0.83,
        is_live=True,
        face_in_db=True,
        consecutive_frames=3,
        bbox=[100, 50, 250, 200],
    )

    def test_publishes_to_correct_topic(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        publisher.publish_event(**self._BASE_KWARGS)
        args, _ = mock_client.publish.call_args
        assert args[0] == TOPIC_EVENTS

    def test_payload_contains_all_required_fields(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        publisher.publish_event(**self._BASE_KWARGS)
        args, _ = mock_client.publish.call_args
        data = json.loads(args[1])
        for field in (
            "decision",
            "identity",
            "similarity",
            "spoof_score",
            "is_live",
            "face_in_db",
            "consecutive_frames",
            "bbox",
            "timestamp",
        ):
            assert field in data, f"missing field: {field}"

    def test_similarity_rounded_to_4_decimals(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        kwargs = {**self._BASE_KWARGS, "similarity": 0.9213456789}
        publisher.publish_event(**kwargs)
        args, _ = mock_client.publish.call_args
        data = json.loads(args[1])
        assert data["similarity"] == 0.9213

    def test_bbox_none_is_serialised(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        kwargs = {**self._BASE_KWARGS, "bbox": None}
        publisher.publish_event(**kwargs)
        args, _ = mock_client.publish.call_args
        data = json.loads(args[1])
        assert data["bbox"] is None


# ---------------------------------------------------------------------------
# publish_status()
# ---------------------------------------------------------------------------


class TestPublishStatus:
    def test_publishes_to_correct_topic_with_qos1(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        publisher.publish_status(door_state="unlocked", last_person="alice")
        args, kwargs = mock_client.publish.call_args
        assert args[0] == TOPIC_STATUS
        assert kwargs.get("qos") == QOS_STATUS or args[2] == QOS_STATUS

    def test_payload_schema(self, publisher: MqttPublisher, mock_client: MagicMock) -> None:
        publisher._connected = True
        publisher.publish_status(door_state="locked", last_person="unknown")
        args, _ = mock_client.publish.call_args
        data = json.loads(args[1])
        assert data["door_state"] == "locked"
        assert data["last_person"] == "unknown"
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# publish_heartbeat()
# ---------------------------------------------------------------------------


class TestPublishHeartbeat:
    _BASE_KWARGS: ClassVar[dict] = dict(
        fps=18.5,
        cpu_temp_c=52.3,
        ram_used_gb=2.1,
        distance_cm=45.0,
        pipeline_stage="IDLE",
        container_uptime_s=120,
    )

    def test_publishes_to_heartbeat_topic(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        publisher.publish_heartbeat(**self._BASE_KWARGS)
        args, _ = mock_client.publish.call_args
        assert args[0] == TOPIC_HEARTBEAT

    def test_payload_contains_all_fields(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        publisher.publish_heartbeat(**self._BASE_KWARGS)
        args, _ = mock_client.publish.call_args
        data = json.loads(args[1])
        for field in (
            "fps",
            "cpu_temp_c",
            "ram_used_gb",
            "distance_cm",
            "pipeline_stage",
            "container_uptime_s",
            "timestamp",
        ):
            assert field in data, f"missing field: {field}"

    def test_fps_rounded_to_2_decimals(
        self, publisher: MqttPublisher, mock_client: MagicMock
    ) -> None:
        publisher._connected = True
        kwargs = {**self._BASE_KWARGS, "fps": 18.5678}
        publisher.publish_heartbeat(**kwargs)
        args, _ = mock_client.publish.call_args
        data = json.loads(args[1])
        assert data["fps"] == 18.57
