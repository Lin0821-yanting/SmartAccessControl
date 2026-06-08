#!/usr/bin/env bash
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
# deploy/entrypoint.sh — container startup script
#
# Responsibilities:
#   1. Compile YOLOv8n-face TRT engine (first run only; cached in volume).
#   2. Verify required ONNX weights and face_db.npy are present.
#   3. Start the orchestrator.
#
# TRT engine compilation is deferred here (not in Dockerfile RUN) because
# CI runners have no GPU; compilation requires the real Jetson hardware.
# The compiled engine is stored in /app/models/engines/ which is backed by
# the named volume "engine-cache" in docker-compose.yml, so it persists
# across container restarts and image updates.

set -euo pipefail

WEIGHTS_DIR=/app/models/weights
ENGINES_DIR=/app/models/engines
CONFIG=/app/configs/config.yaml
HEALTH_FILE=/tmp/healthz

# ── Helper ─────────────────────────────────────────────────────────────────────
log() { echo "[entrypoint] $*"; }
die() { echo "[entrypoint] FATAL: $*" >&2; exit 1; }

# ── 1. Check required ONNX weights ─────────────────────────────────────────────
log "Checking ONNX model weights..."

YOLO_ONNX="${WEIGHTS_DIR}/yolov8n-face.onnx"
MOBILEFACENET_ONNX="${WEIGHTS_DIR}/MobileFaceNet.onnx"   # 大小寫注意
MINIFASNET_ONNX="${WEIGHTS_DIR}/minifasnet_2.7_80x80_GRAY.onnx"

for f in "$YOLO_ONNX" "$MOBILEFACENET_ONNX" "$MINIFASNET_ONNX"; do
    [ -f "$f" ] || die "Required weight not found: $f
Mount the weights directory with:
  -v /path/to/models/weights:/app/models/weights:ro"
done
log "All ONNX weights present."

# ── 2. Compile YOLOv8n-face TRT engine (once, then cached) ────────────────────
YOLO_ENGINE="${ENGINES_DIR}/yolov8n-face-fp16.engine"

if [ -f "$YOLO_ENGINE" ]; then
    log "YOLOv8n-face engine already compiled — using cache."
else
    log "Compiling YOLOv8n-face TRT FP16 engine (≈ 3-5 min on first run)..."
    log "  Source : $YOLO_ONNX"
    log "  Output : $YOLO_ENGINE"

    # Flush page cache to free RAM before TRT compilation.
    # TRT needs ~3 GB; skip if we don't have sudo (non-privileged containers).
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true
    fi

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    trtexec \
        --onnx="${YOLO_ONNX}" \
        --saveEngine="${YOLO_ENGINE}" \
        --fp16 \
        --workspace=2048 \
        --verbose=false

    log "YOLOv8n-face engine compiled successfully."
fi

# MobileFaceNet and MiniFASNet use ONNX Runtime directly — no TRT compilation needed.

# ── 3. Check face_db ────────────────────────────────────────────────────────────
FACE_DB=/app/data/face_db.npy

if [ ! -f "$FACE_DB" ]; then
    log "face_db.npy not found — running enroll.py to build it..."
    python3 /app/scripts/enroll.py || {
        log "WARNING: enroll.py failed or no enrollment photos found."
        log "The system will start but GRANT decisions will not fire."
        log "Mount enrollment photos at /app/data/enrollment/ and restart."
    }
else
    log "face_db.npy present."
fi

# ── 4. Write initial health file ───────────────────────────────────────────────
date +%s > "$HEALTH_FILE"
log "Health file initialised at $HEALTH_FILE"

# ── 5. Start orchestrator ──────────────────────────────────────────────────────
log "Starting Smart Access Control orchestrator..."
log "  Config     : $CONFIG"
log "  MQTT broker: ${MQTT_BROKER}:${MQTT_PORT}"

exec python3 -m src.orchestrator \
    --config "$CONFIG" \
    "$@"
