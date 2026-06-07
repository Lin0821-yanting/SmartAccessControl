#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
"""src/orchestrator.py — top-level integration loop.

Wires the four system layers together in a single run() call:

    Sense  : HC-SR04 distance gate → camera frame
    Process: M1 AI pipeline (FaceDetector → FaceRecognizer → AntiSpoof)
    Decide : DecisionEngine (pure-logic state machine)
    Act    : ActuatorController (GPIO) + MqttPublisher (MQTT)

Threading model
---------------
- Main thread   : camera read → AI pipeline → decide → act → display
- Heartbeat     : daemon thread publishing system metrics at 1 Hz
- ActuatorController internally spawns a servo thread on GRANT so the
  main pipeline loop is not blocked for the 3-second unlock period.

Typical entry point::

    from src.orchestrator import Orchestrator
    orc = Orchestrator.from_config("configs/config.yaml")
    orc.run()
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.actuator_controller import ActuatorController
from src.antispoof.antispoof import AntiSpoof, LivenessResult
from src.decision_engine import LIVENESS_THRESHOLD, Decision, DecisionEngine
from src.detection.detector import FaceDetection, FaceDetector
from src.hc_sr04 import HCSR04
from src.mqtt_publisher import MqttPublisher
from src.recognition.recognizer import FaceRecognizer, RecognitionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GStreamer pipeline helper (same as M1's main.py for consistency)
# ---------------------------------------------------------------------------


def _gstreamer_pipeline(  # pragma: no cover
    sensor_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 20,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


# ---------------------------------------------------------------------------
# Pipeline stage labels (used in heartbeat)
# ---------------------------------------------------------------------------
_STAGE_IDLE = "IDLE"
_STAGE_DETECTING = "DETECTING"
_STAGE_MATCHING = "MATCHING"
_STAGE_DECIDED = "DECIDED"

# HC-SR04 gate distance (cm) — persons closer than this trigger AI pipeline
_GATE_DISTANCE_CM: float = 60.0

# After GRANT, suppress further triggers for this many seconds
_GRANT_COOLDOWN_S: float = 4.0


class Orchestrator:
    """Integrates M1 AI pipeline with M2 actuation and MQTT publishing.

    Parameters
    ----------
    detector, recognizer, antispoof:
        M1 inference modules — injected so Orchestrator is testable.
    engine:
        DecisionEngine instance.
    actuator:
        ActuatorController instance.
    publisher:
        MqttPublisher instance.
    sensor:
        HC-SR04 distance sensor instance.
    display:
        Whether to render an OpenCV window (set False for headless/Docker).
    """

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        antispoof: AntiSpoof,
        engine: DecisionEngine,
        actuator: ActuatorController,
        publisher: MqttPublisher,
        sensor: HCSR04,
        display: bool = False,
    ) -> None:
        """Initialise with all hardware and logic layers injected.

        Parameters
        ----------
        detector:
            M1 YOLOv8n-face TensorRT inference module.
        recognizer:
            M1 MobileFaceNet ONNX identity matching module.
        antispoof:
            M1 MiniFASNet ONNX liveness detection module.
        engine:
            DecisionEngine state machine (pure logic, no hardware).
        actuator:
            ActuatorController driving LED / Buzzer / Servo.
        publisher:
            MqttPublisher for the three access-control topics.
        sensor:
            HC-SR04 distance sensor used as the pipeline gate.
        display:
            Whether to render an OpenCV preview window.
            Set ``False`` for headless / Docker deployment.
        """
        self._detector = detector
        self._recognizer = recognizer
        self._antispoof = antispoof
        self._engine = engine
        self._actuator = actuator
        self._publisher = publisher
        self._sensor = sensor
        self._display = display

        # Door state tracked here so publish_status fires only on change
        self._door_state: str = "locked"
        self._last_person: str = "unknown"

        # Pipeline stage label (written by main thread, read by heartbeat thread)
        self._pipeline_stage: str = _STAGE_IDLE
        self._stage_lock = threading.Lock()

        # FPS measurement
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._fps_window_start: float = time.monotonic()

        # Latest HC-SR04 reading (cm) — written by main loop, read by heartbeat
        self._distance_cm: float = 999.0

        # Process start time for uptime calculation
        self._start_time: float = time.monotonic()

        # Cooldown flag: prevents re-triggering during servo unlock + LED hold
        self._grant_until: float = 0.0

    # ------------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(  # pragma: no cover
        cls,
        config_path: str = "configs/config.yaml",
        display: bool = False,
    ) -> Orchestrator:
        """Construct a fully-wired Orchestrator from a YAML config file.

        Config keys used::

            models:
              yolo:
                engine: models/engines/yolov8n-face-fp16.engine
                input_size: 416
              mobilefacenet:
                engine: models/engines/mobilefacenet-int8.engine
              minifasnet:
                engine: models/weights/minifasnet.onnx
            recognition:
              db_path: data/face_db.npy
              similarity_threshold: 0.85
              confirm_frames: 3
            mqtt:
              broker_host: localhost
              broker_port: 1883
        """
        import os

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        m = cfg["models"]
        r = cfg["recognition"]
        q = cfg.get("mqtt", {})

        detector = FaceDetector(
            engine_path=m["yolo"]["engine"],
            input_size=m["yolo"].get("input_size", 416),
        )
        recognizer = FaceRecognizer(
            onnx_path=m["mobilefacenet"]["engine"],
            db_path=r["db_path"],
            threshold=r["similarity_threshold"],
        )
        antispoof = AntiSpoof(
            onnx_path=m["minifasnet"]["engine"],
            threshold=LIVENESS_THRESHOLD,
        )
        engine = DecisionEngine(
            similarity_threshold=r["similarity_threshold"],
            required_frames=r["confirm_frames"],
        )
        actuator = ActuatorController()
        publisher = MqttPublisher(
            broker_host=os.environ.get("MQTT_BROKER", q.get("broker_host", "localhost")),
            broker_port=int(os.environ.get("MQTT_PORT", q.get("broker_port", 1883))),
        )
        sensor = HCSR04()

        return cls(
            detector=detector,
            recognizer=recognizer,
            antispoof=antispoof,
            engine=engine,
            actuator=actuator,
            publisher=publisher,
            sensor=sensor,
            display=display,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # pragma: no cover
        """Open camera, connect MQTT, and start the main pipeline loop.

        Press Q or Ctrl-C to exit cleanly.
        """
        self._publisher.connect()

        # Publish initial door state
        self._publisher.publish_status(
            door_state=self._door_state,
            last_person=self._last_person,
        )

        # Start 1-Hz heartbeat daemon
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        cap = cv2.VideoCapture(_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("Cannot open CSI camera — check GStreamer pipeline")

        logger.info("Orchestrator: pipeline started")
        print("\n[Orchestrator] Running. Press Q or Ctrl-C to stop.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                self._tick(frame)

                if self._display:
                    cv2.imshow("Smart Access Control", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

        except KeyboardInterrupt:
            print("\n[Orchestrator] Interrupted.")
        finally:
            cap.release()
            if self._display:
                cv2.destroyAllWindows()
            self._actuator.cleanup()
            self._publisher.disconnect()
            logger.info("Orchestrator: shutdown complete")

    # ------------------------------------------------------------------
    # Per-frame logic
    # ------------------------------------------------------------------

    def _tick(self, frame: np.ndarray) -> None:
        """Process a single camera frame end-to-end."""
        # ── Sense: HC-SR04 gate ──────────────────────────────────────────
        self._distance_cm = self._sensor._measure_distance()
        gate_open = self._distance_cm != float("inf") and self._distance_cm < _GATE_DISTANCE_CM

        # Skip AI pipeline if gate closed or still in cooldown
        if not gate_open or time.monotonic() < self._grant_until:
            self._set_stage(_STAGE_IDLE)
            self._engine.ignore()
            self._update_fps()
            return

        # ── Process: Stage 1 — face detection ───────────────────────────
        self._set_stage(_STAGE_DETECTING)
        faces: list[FaceDetection] = self._detector.detect(frame)

        if not faces:
            # Gate fired but no face — HC-SR04 1-second window expired
            decision = self._engine.ignore()
            self._publish_event_from(
                decision=decision,
                identity="unknown",
                similarity=0.0,
                spoof_score=0.0,
                is_live=False,
                face_in_db=False,
                bbox=None,
            )
            self._update_fps()
            return

        # Process the highest-confidence face only
        face = faces[0]
        bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]

        # ── Process: Stage 2 & 3 — recognition + anti-spoof ─────────────
        self._set_stage(_STAGE_MATCHING)
        recog: RecognitionResult = self._recognizer.match(face.crop)
        liveness: LivenessResult = self._antispoof.predict(face.crop)

        anti_spoof_pass = liveness.is_live
        face_in_db = recog.authorized  # True when sim >= threshold

        # ── Decide: DecisionEngine ───────────────────────────────────────
        self._set_stage(_STAGE_DECIDED)
        decision: Decision = self._engine.evaluate(
            similarity=recog.similarity,
            anti_spoof_pass=anti_spoof_pass,
            face_in_db=face_in_db,
        )

        # ── Log ──────────────────────────────────────────────────────────
        logger.info(
            "DECISION=%s  identity=%s  sim=%.3f  live=%.3f  frames=%d",
            decision.name,
            recog.name,
            recog.similarity,
            liveness.score,
            self._engine.consecutive_frames,
        )

        # ── Publish: lab/access/events ───────────────────────────────────
        self._publish_event_from(
            decision=decision,
            identity=recog.name,
            similarity=recog.similarity,
            spoof_score=liveness.score,
            is_live=anti_spoof_pass,
            face_in_db=face_in_db,
            bbox=bbox,
        )

        # ── Act: actuator + door-state publish ───────────────────────────
        self._act(decision, identity=recog.name)
        self._update_fps()

    # ------------------------------------------------------------------
    # Act helpers
    # ------------------------------------------------------------------

    def _act(self, decision: Decision, identity: str) -> None:
        """Drive actuators and publish status when door state changes."""
        if decision == Decision.GRANT:
            self._grant_until = time.monotonic() + _GRANT_COOLDOWN_S
            # Actuator: servo unlock + green LED (non-blocking internally)
            threading.Thread(target=self._actuator.grant_access, daemon=True).start()
            # Status: unlocked
            self._set_door_state("unlocked", identity)
            # Schedule auto-relock status publish after servo returns
            threading.Timer(
                _GRANT_COOLDOWN_S,
                self._set_door_state,
                args=("locked", identity),
            ).start()

        elif decision == Decision.DENY:
            threading.Thread(target=self._actuator.deny_access, daemon=True).start()

        elif decision == Decision.UNKNOWN:
            threading.Thread(target=self._actuator.alert_unknown, daemon=False).start()

        elif decision == Decision.SPOOF:
            threading.Thread(target=self._actuator.alert_spoof, daemon=False).start()

        # IGNORE → no actuator action

    def _set_door_state(self, state: str, person: str) -> None:
        """Update tracked door state and publish only if it changed."""
        if state != self._door_state:
            self._door_state = state
            self._last_person = person
            self._publisher.publish_status(
                door_state=state,
                last_person=person,
            )
            logger.info("Door state → %s (person=%s)", state, person)

    # ------------------------------------------------------------------
    # Publish helper
    # ------------------------------------------------------------------

    def _publish_event_from(
        self,
        *,
        decision: Decision,
        identity: str,
        similarity: float,
        spoof_score: float,
        is_live: bool,
        face_in_db: bool,
        bbox: list[int] | None,
    ) -> None:
        """Build and publish a lab/access/events payload."""
        self._publisher.publish_event(
            decision=decision.name,
            identity=identity,
            similarity=similarity,
            spoof_score=spoof_score,
            is_live=is_live,
            face_in_db=face_in_db,
            consecutive_frames=self._engine.consecutive_frames,
            bbox=bbox,
        )

    # ------------------------------------------------------------------
    # Heartbeat daemon
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Publish system health metrics at 1 Hz (runs in daemon thread)."""
        while True:
            time.sleep(1.0)
            try:
                self._publisher.publish_heartbeat(
                    fps=self._fps,
                    cpu_temp_c=_read_cpu_temp(),
                    ram_used_gb=_read_ram_gb(),
                    distance_cm=self._distance_cm if self._distance_cm else -1.0,
                    pipeline_stage=self._get_stage(),
                    container_uptime_s=int(time.monotonic() - self._start_time),
                )
                Path("/tmp/healthz").write_text(str(time.time()))
            except Exception:
                logger.exception("Heartbeat publish failed")

    # ------------------------------------------------------------------
    # FPS tracking
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_window_start = now

    # ------------------------------------------------------------------
    # Pipeline stage (thread-safe string access)
    # ------------------------------------------------------------------

    def _set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._pipeline_stage = stage

    def _get_stage(self) -> str:
        with self._stage_lock:
            return self._pipeline_stage


# ---------------------------------------------------------------------------
# System metric helpers
# ---------------------------------------------------------------------------


def _read_cpu_temp() -> float:
    """Read CPU temperature from sysfs (°C). Returns -1 on failure."""
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        return int(Path(path).read_text().strip()) / 1000.0
    except OSError:
        return -1.0


def _read_ram_gb() -> float:
    """Read used RAM in GB from /proc/meminfo. Returns -1 on failure."""
    try:
        meminfo: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, val = line.partition(":")
            meminfo[key.strip()] = int(val.split()[0])
        total = meminfo.get("MemTotal", 0)
        avail = meminfo.get("MemAvailable", 0)
        return round((total - avail) / (1024**2), 3)
    except (OSError, ValueError, KeyError):
        return -1.0


# ---------------------------------------------------------------------------
# CLI entry point (convenience — main.py should normally be used)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Smart Access Control Orchestrator")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    orc = Orchestrator.from_config(args.config, display=args.display)
    orc.run()
