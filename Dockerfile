# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
#
# Multi-stage Dockerfile for Smart Access Control on Jetson Orin Nano.
#
# Stage 1 (builder) — installs additional pip packages on top of the
#   dustynv base.  The dustynv image ships torch, torchvision, cv2,
#   onnxruntime, and tensorrt already; we only add paho-mqtt, gpiod,
#   and pyyaml here.
#
# Stage 2 (runtime) — copies only the installed packages + app code
#   into a clean copy of the same base, keeping the final image lean.
#
# TRT engine compilation is intentionally deferred to deploy/entrypoint.sh
# so the image can be built on an x86 CI runner (no GPU available at
# build time).  The compiled engine is stored in a Docker volume
# (engine-cache) and reused across container restarts.
#
# Build (CI — ubuntu-latest with QEMU):
#   docker buildx build --platform linux/arm64 -t ghcr.io/<owner>/<repo>:sha-<sha> --push .
#
# Run (Jetson):
#   docker compose -f deploy/docker-compose.yml up -d

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM --platform=linux/arm64 dustynv/pytorch:2.7-r36.4.0 AS builder

# Install only what is NOT already in the dustynv base image.
# --break-system-packages is required because the base image uses
# the system Python (not a venv) and newer pip refuses to install
# into it without this flag.
RUN pip install \
        "paho-mqtt>=2.0" \
        "gpiod>=2.4.2" \
        "pyyaml>=6.0" \
    --break-system-packages \
    --no-cache-dir

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/arm64 dustynv/pytorch:2.7-r36.4.0 AS runtime

LABEL org.opencontainers.image.source="https://github.com/Lin0821-yanting/SmartAccessControl"
LABEL org.opencontainers.image.description="Smart Laboratory Access Control — Jetson Orin Nano"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy pip-installed packages from the builder stage.
COPY --from=builder \
    /usr/local/lib/python3.10/dist-packages \
    /usr/local/lib/python3.10/dist-packages

# ── Application source code ───────────────────────────────────────────────────
COPY src/        ./src/
COPY configs/    ./configs/

# ONNX weights are baked into the image so the container is self-contained.
# TRT engines are NOT copied here — they are compiled by entrypoint.sh at
# first startup and cached in a named volume (engine-cache).
# COPY models/weights/ ./models/weights/

# Enrollment face photos are NOT baked in; they are mounted as a volume at
# runtime so new faces can be enrolled without rebuilding the image.
# The scripts/ directory IS included so enroll.py can be run inside the container.
COPY scripts/    ./scripts/

# ── Runtime directories ────────────────────────────────────────────────────────
# These will be bind-mounted or named volumes at compose time, but we create
# them here so the container starts cleanly even without a mount.
RUN mkdir -p /app/models/engines \
             /app/data/enrollment \
             /tmp

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
# MQTT_BROKER is overridden by docker-compose to the service name "mosquitto".
# The default "localhost" allows running the container standalone for debugging.
ENV MQTT_BROKER=mosquitto
ENV MQTT_PORT=1883
# Suppress GTK/EGL errors in headless SSH sessions (same fix as bare-Jetson dev).
ENV DISPLAY=

ENTRYPOINT ["/entrypoint.sh"]
