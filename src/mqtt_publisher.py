#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""src/mqtt_publisher.py — MQTT publish façade for the access-control system.

Encapsulates paho-mqtt and provides three typed publish methods that map
directly onto the three topics defined in proposal §4.7:

    Topic                   Method              Trigger
    ──────────────────────────────────────────────────────────────
    lab/access/events       publish_event()     every DecisionEngine output
    lab/access/status       publish_status()    door-state change only
    lab/access/heartbeat    publish_heartbeat() 1 Hz system health

All payloads are JSON-encoded dicts. If the client is not yet connected,
publish methods return False without raising.

Usage::

    pub = MqttPublisher()
    pub.connect()
    pub.publish_event(...)
    pub.disconnect()
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic constants — import these in other modules instead of hardcoding strings
# ---------------------------------------------------------------------------
TOPIC_EVENTS: str = "lab/access/events"
TOPIC_STATUS: str = "lab/access/status"
TOPIC_HEARTBEAT: str = "lab/access/heartbeat"

# ---------------------------------------------------------------------------
# Connection defaults
# ---------------------------------------------------------------------------
DEFAULT_BROKER_HOST: str = "localhost"
DEFAULT_BROKER_PORT: int = 1883
DEFAULT_KEEPALIVE_SEC: int = 60

# QoS levels per topic (matches proposal rationale)
QOS_EVENTS: int = 0      # per-decision; tolerate drops, latency matters
QOS_STATUS: int = 1      # door state; must be delivered
QOS_HEARTBEAT: int = 0   # 1 Hz health; stale data is fine


class MqttPublisher:
    """Paho-MQTT wrapper with typed publish methods for each access-control topic.

    Parameters
    ----------
    broker_host:
        Hostname or IP of the MQTT broker (default ``"localhost"``).
    broker_port:
        TCP port of the broker (default ``1883``).
    client_id:
        MQTT client identifier.  Defaults to ``"jetson-access-control"``.
    client_factory:
        Injectable factory for the paho client — used in unit tests to
        inject a mock without patching the module.
    """

    def __init__(
        self,
        broker_host: str = DEFAULT_BROKER_HOST,
        broker_port: int = DEFAULT_BROKER_PORT,
        client_id: str = "jetson-access-control",
        client_factory=None,
    ) -> None:
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._connected = False

        factory = client_factory or (lambda: mqtt.Client(
            client_id=client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        ))
        self._client: mqtt.Client = factory()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to broker and start background network loop."""
        self._client.connect(
            self._broker_host,
            self._broker_port,
            keepalive=DEFAULT_KEEPALIVE_SEC,
        )
        self._client.loop_start()
        # Give on_connect callback time to fire
        _deadline = time.monotonic() + 5.0
        while not self._connected and time.monotonic() < _deadline:
            time.sleep(0.05)
        if not self._connected:
            logger.warning("MqttPublisher: broker did not ACK connect within 5 s")

    def disconnect(self) -> None:
        """Gracefully stop the loop and disconnect."""
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False
        logger.info("MqttPublisher: disconnected")

    @property
    def connected(self) -> bool:
        """True when the broker connection is established."""
        return self._connected

    # ------------------------------------------------------------------
    # Typed publish methods
    # ------------------------------------------------------------------

    def publish_event(
        self,
        *,
        decision: str,
        identity: str,
        similarity: float,
        spoof_score: float,
        is_live: bool,
        face_in_db: bool,
        consecutive_frames: int,
        bbox: list[int] | None = None,
    ) -> bool:
        """Publish one access-decision event to ``lab/access/events``.

        Called once per DecisionEngine.evaluate() output (every frame that
        produces a non-trivial decision).

        Parameters
        ----------
        decision:
            String form of the Decision enum: "GRANT" / "DENY" /
            "UNKNOWN" / "SPOOF" / "IGNORE".
        identity:
            Matched identity name from MobileFaceNet, or ``"unknown"``.
        similarity:
            Stage 4 cosine-similarity score (0.0 – 1.0).
        spoof_score:
            Stage 3 MiniFASNet real-face probability (higher = more live).
        is_live:
            True when spoof_score >= liveness threshold.
        face_in_db:
            True when an enrolled match was found above the similarity threshold.
        consecutive_frames:
            Current value of DecisionEngine.consecutive_frames at the time
            of this evaluation.
        bbox:
            Stage 1 face bounding box ``[x1, y1, x2, y2]`` as integers, or
            None when no face was detected.
        """
        payload: dict[str, Any] = {
            "decision": decision,
            "identity": identity,
            "similarity": round(similarity, 4),
            "spoof_score": round(spoof_score, 4),
            "is_live": is_live,
            "face_in_db": face_in_db,
            "consecutive_frames": consecutive_frames,
            "bbox": bbox,
            "timestamp": _now_iso(),
        }
        return self.publish(TOPIC_EVENTS, payload, qos=QOS_EVENTS)

    def publish_status(
        self,
        *,
        door_state: str,
        last_person: str,
    ) -> bool:
        """Publish a door-state change to ``lab/access/status``.

        Should be called only when the door state actually changes
        (locked → unlocked on GRANT; unlocked → locked on auto-relock).

        Parameters
        ----------
        door_state:
            ``"locked"`` or ``"unlocked"``.
        last_person:
            Identity of the last person who was GRANTed access,
            or ``"unknown"`` when initialising.
        """
        payload: dict[str, Any] = {
            "door_state": door_state,
            "last_person": last_person,
            "timestamp": _now_iso(),
        }
        return self.publish(TOPIC_STATUS, payload, qos=QOS_STATUS)

    def publish_heartbeat(
        self,
        *,
        fps: float,
        cpu_temp_c: float,
        ram_used_gb: float,
        distance_cm: float,
        pipeline_stage: str,
        container_uptime_s: int,
    ) -> bool:
        """Publish a 1-Hz system health snapshot to ``lab/access/heartbeat``.

        Parameters
        ----------
        fps:
            Measured inference frames per second (rolling 1-s window).
        cpu_temp_c:
            CPU temperature in °C from /sys/class/thermal.
        ram_used_gb:
            RAM currently in use (GB).
        distance_cm:
            Latest HC-SR04 distance reading in centimetres.
        pipeline_stage:
            Current pipeline phase: ``"IDLE"`` / ``"DETECTING"`` /
            ``"MATCHING"`` / ``"DECIDED"``.
        container_uptime_s:
            Seconds since the container process started.
        """
        payload: dict[str, Any] = {
            "fps": round(fps, 2),
            "cpu_temp_c": round(cpu_temp_c, 1),
            "ram_used_gb": round(ram_used_gb, 3),
            "distance_cm": round(distance_cm, 1),
            "pipeline_stage": pipeline_stage,
            "container_uptime_s": container_uptime_s,
            "timestamp": _now_iso(),
        }
        return self.publish(TOPIC_HEARTBEAT, payload, qos=QOS_HEARTBEAT)

    # ------------------------------------------------------------------
    # Low-level publish (also useful in tests)
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: dict | str,
        qos: int = 0,
    ) -> bool:
        """JSON-encode *payload* and publish to *topic*.

        Returns False (without raising) when not connected.
        """
        if not self._connected:
            logger.debug("MqttPublisher: not connected — skipping publish to %s", topic)
            return False

        body = payload if isinstance(payload, str) else json.dumps(payload)
        info = self._client.publish(topic, body, qos=qos)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning("MqttPublisher: publish failed rc=%d topic=%s", info.rc, topic)
            return False
        return True

    # ------------------------------------------------------------------
    # Paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, connect_flags, rc, properties=None) -> None:  # noqa: ANN001
        if rc == 0:
            self._connected = True
            logger.info("MqttPublisher: connected to %s:%d", self._broker_host, self._broker_port)
        else:
            logger.error("MqttPublisher: connect failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, disconnect_flags, rc, properties=None) -> None:  # noqa: ANN001
        self._connected = False
        if rc != 0:
            logger.warning("MqttPublisher: unexpected disconnect rc=%d — paho will retry", rc)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
