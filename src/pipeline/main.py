#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai, Yanting Lin
# Tatung University I4210 AI實務專題
"""src/pipeline/main.py — 智慧門禁系統主程式。

整合 YOLOv8n-face + MobileFaceNet + MiniFASNet + HC-SR04 + ActuatorController
透過 IMX219 CSI 相機進行即時人臉辨識與活體偵測。

決策邏輯（三條件同時滿足才授權）：
  1. cosine similarity >= 0.85
  2. 連續 3 幀匹配同一人
  3. 活體偵測通過（is_live = True）

MQTT 格式與 orchestrator.py 一致：
  lab/access/events    — 每幀決策結果
  lab/access/status    — 門鎖狀態變更
  lab/access/heartbeat — 1 Hz 系統健康

Usage:
    pdm run python src/pipeline/main.py
    pdm run python src/pipeline/main.py --no-display   # headless 模式
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.actuator_controller import ActuatorController
from src.antispoof.antispoof import AntiSpoof
from src.detection.detector import FaceDetector
from src.hc_sr04 import HcSr04
from src.mqtt_publisher import MqttPublisher
from src.recognition.recognizer import FaceRecognizer

# ---------------------------------------------------------------------------
# Pipeline stage labels (matches orchestrator.py)
# ---------------------------------------------------------------------------
_STAGE_IDLE = "IDLE"
_STAGE_DETECTING = "DETECTING"
_STAGE_MATCHING = "MATCHING"
_STAGE_DECIDED = "DECIDED"

# HC-SR04 gate distance (cm). Wider gate lets the user stand further back so
# the face occupies less of the frame (more background context) — which the
# anti-spoof model scores far more reliably than a close-up, face-filling frame.
_GATE_DISTANCE_CM: float = 100.0

# After GRANT, suppress further triggers for this many seconds
_GRANT_COOLDOWN_S: float = 4.5

# After a DENY/UNKNOWN/SPOOF alert, suppress further alert actuator triggers
# for this many seconds. Prevents the buzzer/LED from firing on every frame
# while the model keeps emitting UNKNOWN for the same person in front of the
# camera (otherwise it re-triggers ~every 2 s and sounds non-stop).
_ALERT_COOLDOWN_S: float = 5.0

# ── Liveness (anti-spoof) live-crop tuning ───────────────────────────────
# MiniFASNet relies on the background/context around the face to judge
# liveness. The detector's tight 20% crop starves it of context, so the
# real person flickers as SPOOF. Expand the bbox by this scale (≈2.7×) and
# feed THAT to the anti-spoof model only (recognition keeps the tight crop).
# Tune live with: LIVENESS_SCALE=3.2 pdm run python src/pipeline/main.py ...
_LIVENESS_CROP_SCALE: float = float(os.environ.get("LIVENESS_SCALE", "2.7"))
# Which framing to feed the anti-spoof model. "full" = the whole camera frame
# (matches how the enrollment images were captured → most stable real scores,
# median ~0.76 vs ~0.19 for a tight crop); "crop" = enlarged face crop.
_LIVENESS_FRAME: str = os.environ.get("LIVENESS_FRAME", "full")
# Liveness pass threshold applied to the model's real-face probability.
# The model's built-in 0.6 is too strict for our camera/distance; the real
# person sits ~0.25–0.5 while a photo sits lower, so the separating threshold
# lives in that gap. Tune live with LIVENESS_THRESHOLD=0.3.
_LIVENESS_THRESHOLD: float = float(os.environ.get("LIVENESS_THRESHOLD", "0.3"))
# Require this many CONSECUTIVE liveness-fail frames before declaring SPOOF.
# A real face dips below threshold only sporadically; a real photo/screen fails
# persistently. This absorbs the flicker so the buzzer fires only on a genuine,
# sustained spoof — while keeping spoof rejection. Tune live with SPOOF_FRAMES=4.
_SPOOF_CONFIRM_FRAMES: int = int(os.environ.get("SPOOF_FRAMES", "5"))


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# GStreamer pipeline
# ---------------------------------------------------------------------------
def gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 20,
    flip_method: int = 0,
) -> str:
    """Build GStreamer pipeline string for CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


# ---------------------------------------------------------------------------
# System metric helpers (same as orchestrator.py)
# ---------------------------------------------------------------------------
def _read_cpu_temp() -> float:
    """Read CPU temperature from sysfs (°C). Returns -1 on failure."""
    path = "/sys/class/thermal/thermal_zone0/temp"
    try:
        return int(Path(path).read_text().strip()) / 1000.0
    except OSError:
        return -1.0


def _expand_crop(
    frame: np.ndarray, bbox: np.ndarray, scale: float
) -> np.ndarray:
    """Return a square crop centred on *bbox*, enlarged by *scale*.

    The anti-spoof model needs context around the face; the detector's tight
    crop is too small. Expanding to ~2.7× the face box and clamping to the
    frame restores the background cues MiniFASNet was trained on.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(w, int(cx + side / 2))
    ny2 = min(h, int(cy + side / 2))
    return frame[ny1:ny2, nx1:nx2].copy()


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
# Temporal voter (3-frame consistency check)
# ---------------------------------------------------------------------------
class TemporalVoter:
    """連續 N 幀匹配同一人才授權，防止單幀照片攻擊。"""

    def __init__(self, required_frames: int = 3) -> None:
        self.required = required_frames
        self.buffer: deque = deque(maxlen=required_frames)

    def vote(self, name: str) -> tuple[bool, str]:
        """加入一幀辨識結果，回傳是否達成授權條件。"""
        self.buffer.append(name)
        if len(self.buffer) == self.required:
            names = list(self.buffer)
            if len(set(names)) == 1 and names[0] != "unknown":
                return True, names[0]
        return False, "unknown"

    def reset(self) -> None:
        """清空 buffer。"""
        self.buffer.clear()

    @property
    def consecutive_frames(self) -> int:
        """目前 buffer 裡有幾幀。"""
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Draw overlay
# ---------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    faces: list,
    decisions: list,
    fps: float,
    distance_cm: float,
    pipeline_stage: str,
) -> np.ndarray:
    """在影像上繪製偵測結果與系統狀態。"""
    vis = frame.copy()

    for face, (decision, name, similarity, is_live) in zip(faces, decisions):
        x1, y1, x2, y2 = face.bbox.astype(int)
        granted = decision == "GRANT"
        color = (0, 255, 0) if granted else (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"{name} ({face.confidence:.2f})"
        cv2.putText(vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for kp in face.keypoints:
            cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

        status = f"{decision} sim={similarity:.3f} live={is_live}"
        cv2.putText(vis, status, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(vis, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(vis, f"Dist: {distance_cm:.1f}cm", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(vis, f"Stage: {pipeline_stage}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    return vis


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(config: dict, display: bool = True) -> None:
    """主要 pipeline 迴圈：Sense → Process → Decide → Act → Publish。"""

    cfg_models = config["models"]
    cfg_recog = config["recognition"]
    cfg_mqtt = config.get("mqtt", {})

    # ── MQTT publisher (same interface as orchestrator.py) ────────────────
    publisher = MqttPublisher(
        broker_host=os.environ.get(
            "MQTT_BROKER", cfg_mqtt.get("broker_host", "localhost")
        ),
        broker_port=int(os.environ.get(
            "MQTT_PORT", cfg_mqtt.get("broker_port", 1883)
        )),
    )
    publisher.connect()
    publisher.publish_status(door_state="locked", last_person="unknown")

    # ── Hardware ──────────────────────────────────────────────────────────
    sensor = HcSr04()
    actuator = ActuatorController()

    # ── AI models ─────────────────────────────────────────────────────────
    detector = FaceDetector(
        engine_path=cfg_models["yolo"]["engine"],
        conf_threshold=0.5,
        input_size=cfg_models["yolo"]["input_size"],
    )
    recognizer = FaceRecognizer(
        onnx_path=cfg_models["mobilefacenet"]["engine"],
        db_path=cfg_recog["db_path"],
        threshold=cfg_recog["similarity_threshold"],
    )
    antispoof = AntiSpoof(
        onnx_path=cfg_models["minifasnet"]["engine"],
    )

    voter = TemporalVoter(required_frames=cfg_recog["confirm_frames"])

    # ── State ─────────────────────────────────────────────────────────────
    door_state = "locked"
    last_person = "unknown"
    grant_until = 0.0          # GRANT cooldown timestamp
    alert_until = 0.0          # DENY/UNKNOWN/SPOOF alert cooldown timestamp
    spoof_streak = 0           # consecutive liveness-fail frames
    distance_cm = 999.0
    pipeline_stage = _STAGE_IDLE
    start_time = time.monotonic()

    # FPS
    frame_count = 0
    fps = 0.0
    fps_window_start = time.monotonic()

    # ── Heartbeat daemon (1 Hz, same payload as orchestrator.py) ─────────
    def heartbeat_loop() -> None:
        while True:
            time.sleep(1.0)
            try:
                publisher.publish_heartbeat(
                    fps=fps,
                    cpu_temp_c=_read_cpu_temp(),
                    ram_used_gb=_read_ram_gb(),
                    distance_cm=distance_cm,
                    pipeline_stage=pipeline_stage,
                    container_uptime_s=int(time.monotonic() - start_time),
                )
            except Exception:
                pass

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("無法開啟 CSI 相機")

    print(
        f"[Pipeline] gate={_GATE_DISTANCE_CM:.0f}cm  "
        f"liveness frame={_LIVENESS_FRAME} thr={_LIVENESS_THRESHOLD}  "
        f"spoof_confirm={_SPOOF_CONFIRM_FRAMES} frames"
    )
    print("\n[Pipeline] 啟動！按 Q 或 Ctrl+C 結束。\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # ── Sense: HC-SR04 gate ───────────────────────────────────────
            distance_cm = sensor._measure_distance()
            gate_open = (
                distance_cm != float("inf")
                and distance_cm < _GATE_DISTANCE_CM
            )

            # ── Gate closed or in cooldown → IDLE ────────────────────────
            if not gate_open or time.monotonic() < grant_until:
                pipeline_stage = _STAGE_IDLE

                # FPS update
                frame_count += 1
                now = time.monotonic()
                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_window_start = now

                if display:
                    vis = draw_overlay(frame, [], [], fps, distance_cm, pipeline_stage)
                    cv2.imshow("Smart Access Control", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            # ── Stage 1: Face detection ───────────────────────────────────
            pipeline_stage = _STAGE_DETECTING
            faces = detector.detect(frame)

            if not faces:
                # Gate triggered but no face
                publisher.publish_event(
                    decision="IGNORE",
                    identity="unknown",
                    similarity=0.0,
                    spoof_score=0.0,
                    is_live=False,
                    face_in_db=False,
                    consecutive_frames=voter.consecutive_frames,
                    bbox=None,
                )
                frame_count += 1
                now = time.monotonic()
                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_window_start = now
                if display:
                    vis = draw_overlay(frame, [], [], fps, distance_cm, pipeline_stage)
                    cv2.imshow("Smart Access Control", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            # Process highest-confidence face only
            face = faces[0]
            bbox = face.bbox.astype(int).tolist()

            # ── Stage 2 & 3: Recognition + anti-spoof ────────────────────
            pipeline_stage = _STAGE_MATCHING
            recog = recognizer.match(face.crop)
            # Anti-spoof: feed the framing chosen by LIVENESS_FRAME. The full
            # frame matches the enrollment capture distribution and is far more
            # stable than the detector's tight crop. Recognition keeps the
            # tight crop above.
            if _LIVENESS_FRAME == "crop":
                live_input = _expand_crop(frame, face.bbox, _LIVENESS_CROP_SCALE)
            else:
                live_input = frame
            liveness = antispoof.predict(live_input)

            anti_spoof_pass = liveness.score >= _LIVENESS_THRESHOLD
            face_in_db = recog.authorized

            # ── Stage 4: Decision ─────────────────────────────────────────
            pipeline_stage = _STAGE_DECIDED

            if not anti_spoof_pass:
                # Liveness failed THIS frame. Require several CONSECUTIVE fail
                # frames before declaring SPOOF: a real face only dips below
                # threshold sporadically, whereas a real photo/screen fails
                # persistently. Below the streak we emit IGNORE (no actuator).
                spoof_streak += 1
                voter.reset()
                decision = "SPOOF" if spoof_streak >= _SPOOF_CONFIRM_FRAMES else "IGNORE"
            else:
                spoof_streak = 0
                if not face_in_db:
                    # UNKNOWN
                    decision = "UNKNOWN"
                    voter.reset()
                else:
                    # Check temporal consistency
                    authorized, matched_name = voter.vote(recog.name)
                    decision = "GRANT" if authorized else "DENY"

            identity = recog.name if recog.name else "unknown"

            print(
                f"[DECISION] {decision:<7} identity={identity:<10} "
                f"sim={recog.similarity:.3f} live={liveness.score:.3f} "
                f"frames={voter.consecutive_frames}/{cfg_recog['confirm_frames']}"
            )

            # ── Publish event (same format as orchestrator.py) ────────────
            publisher.publish_event(
                decision=decision,
                identity=identity,
                similarity=recog.similarity,
                spoof_score=liveness.score,
                is_live=anti_spoof_pass,
                face_in_db=face_in_db,
                consecutive_frames=voter.consecutive_frames,
                bbox=bbox,
            )

            # ── Act: actuator + door state ────────────────────────────────
            if decision == "GRANT":
                grant_until = time.monotonic() + _GRANT_COOLDOWN_S
                threading.Thread(
                    target=actuator.grant_access, daemon=False
                ).start()
                # Publish status: unlocked
                if door_state != "unlocked":
                    door_state = "unlocked"
                    last_person = identity
                    publisher.publish_status(
                        door_state="unlocked", last_person=identity
                    )
                voter.reset()
                # Schedule auto-relock status
                def _relock(name: str = identity) -> None:
                    time.sleep(_GRANT_COOLDOWN_S)
                    nonlocal door_state
                    door_state = "locked"
                    publisher.publish_status(
                        door_state="locked", last_person=name
                    )
                threading.Thread(target=_relock, daemon=True).start()

            elif decision in ("DENY", "UNKNOWN", "SPOOF") and time.monotonic() >= alert_until:
                # Debounce DENY/UNKNOWN/SPOOF: fire the actuator at most once
                # per _ALERT_COOLDOWN_S. Without this, a model that keeps
                # emitting UNKNOWN for the same person re-triggers the LED/buzzer
                # every ~2 s and sounds like a non-stop beep.
                alert_until = time.monotonic() + _ALERT_COOLDOWN_S
                print(
                    f"[ACT] {decision} alert fired — "
                    f"suppressed for {_ALERT_COOLDOWN_S:.1f}s"
                )

                if decision == "SPOOF":
                    # Red LED + buzzer (demonstrates spoof rejection)
                    threading.Thread(
                        target=actuator.alert_spoof, daemon=False
                    ).start()
                else:
                    # DENY / UNKNOWN — red LED only, no buzzer
                    threading.Thread(
                        target=actuator.deny_access, daemon=False
                    ).start()

            # ── FPS update ────────────────────────────────────────────────
            frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_window_start = now

            # ── Display ───────────────────────────────────────────────────
            if display:
                decisions_overlay = [
                    (decision, identity, recog.similarity, anti_spoof_pass)
                ]
                vis = draw_overlay(
                    frame, [face], decisions_overlay, fps,
                    distance_cm, pipeline_stage
                )
                cv2.imshow("Smart Access Control", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\n[Pipeline] 使用者中斷。")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        actuator.cleanup()
        publisher.disconnect()
        print("[Pipeline] 結束。")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="智慧門禁系統主程式")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="config.yaml 路徑",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="headless 模式（不顯示畫面）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, display=not args.no_display)


if __name__ == "__main__":
    main()