# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
#
# Multi-stage Dockerfile for Smart Access Control on Jetson Orin Nano.
#
# Stage 1 (builder) — installs all project deps via pdm export from pyproject.toml.
#   pyproject.toml is the single source of truth (Lab10/HW6 standard).
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

WORKDIR /build

# Copy dependency files (pyproject.toml is the single source of truth).
# pdm.lock pins exact versions to ensure reproducible builds.
COPY pyproject.toml pdm.lock ./

# Use pdm export to generate requirements.txt from pyproject.toml.
# This is the Lab10/HW6 standard: pyproject.toml → pdm export → pip install.
# After export, uninstall pdm so it doesn't ship in the final image.
#
# numpy==1.26.4 is pinned explicitly because ultralytics' transitive dep
# (matplotlib) requests numpy>=2, which would clobber dustynv's numpy 1.x
# and break torch's C extensions (compiled against numpy 1.x ABI).
RUN pip install pdm \
        --index-url https://pypi.org/simple \
        --break-system-packages \
        --no-cache-dir && \
    pdm export \
        --no-hashes \
        --without dev,quality \
        --output requirements.txt && \
    pip uninstall -y pdm && \
    pip install \
        -r requirements.txt \
        "numpy==1.26.4" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
    --break-system-packages \
    --no-cache-dir && \
    rm -f requirements.txt pyproject.toml pdm.lock

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/arm64 dustynv/pytorch:2.7-r36.4.0 AS runtime

LABEL org.opencontainers.image.source="https://github.com/Lin0821-yanting/SmartAccessControl"
LABEL org.opencontainers.image.description="Smart Laboratory Access Control — Jetson Orin Nano"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system libraries required by cv2 (opencv-python-headless).
# libGL.so.1 is needed even by the headless variant at import time.
# libglib2.0-0 provides libgobject and other GLib dependencies cv2 uses.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-opencv \
        python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy pip-installed packages from the builder stage.
COPY --from=builder \
    /usr/local/lib/python3.10/dist-packages \
    /usr/local/lib/python3.10/dist-packages

# ── Application source code ───────────────────────────────────────────────────
COPY src/        ./src/
COPY configs/    ./configs/
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
# The default "mosquitto" works with the compose stack; use "localhost" for
# standalone docker run testing with a local broker.
ENV MQTT_BROKER=mosquitto
ENV MQTT_PORT=1883
# Suppress GTK/EGL errors in headless SSH sessions.
ENV DISPLAY=

ENTRYPOINT ["/entrypoint.sh"]