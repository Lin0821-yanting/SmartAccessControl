#!/usr/bin/env bash
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
# deploy/healthcheck.sh — poll container health, require 3 consecutive successes
#
# Usage (called by deploy.sh after docker compose up):
#   bash deploy/healthcheck.sh [max_wait_seconds]   # default: 60
#
# Exit codes:
#   0 — 3 consecutive successes observed within the timeout
#   1 — timeout before 3 consecutive successes

set -euo pipefail

MAX_WAIT=${1:-60}        # total seconds to wait before giving up
POLL_INTERVAL=5          # seconds between checks
REQUIRED_SUCCESSES=3     # consecutive successes required
STALE_HEALTH_SECS=30     # health file must be newer than this

HEALTH_FILE=/tmp/smartaccess-healthz   # written inside the container
CONTAINER_NAME=deploy-access-control-1 # default docker compose container name

consecutive=0
elapsed=0

echo "[healthcheck] Waiting for container to become healthy..."
echo "[healthcheck] Required: ${REQUIRED_SUCCESSES} consecutive successes within ${MAX_WAIT}s"

while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    # ── Check 1: container is running ──────────────────────────────────────
    state=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "notfound")
    if [ "$state" != "running" ]; then
        echo "[healthcheck] Container '$CONTAINER_NAME' is $state (not running)"
        consecutive=0
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
        continue
    fi

    # ── Check 2: health file updated recently ──────────────────────────────
    # orchestrator._heartbeat_loop() writes epoch seconds to /tmp/healthz
    # inside the container once per second.
    last_beat=$(docker exec "$CONTAINER_NAME" \
        cat /tmp/healthz 2>/dev/null | tr -d '.' | cut -c1-10 || echo "0")
    now=$(date +%s)
    age=$((now - last_beat))

    if [ "$age" -gt "$STALE_HEALTH_SECS" ]; then
        echo "[healthcheck] Health file is ${age}s old (stale threshold: ${STALE_HEALTH_SECS}s)"
        consecutive=0
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
        continue
    fi

    # ── Check 3: MQTT broker reachable from container ──────────────────────
    if ! docker exec "$CONTAINER_NAME" \
        python3 -c "import socket; socket.create_connection(('mosquitto', 1883), 3).close()" \
        2>/dev/null; then
        echo "[healthcheck] MQTT broker not reachable from container"
        consecutive=0
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
        continue
    fi

    # ── All checks passed ─────────────────────────────────────────────────
    consecutive=$((consecutive + 1))
    echo "[healthcheck] OK (${consecutive}/${REQUIRED_SUCCESSES}) — health=${age}s ago"

    if [ "$consecutive" -ge "$REQUIRED_SUCCESSES" ]; then
        echo "[healthcheck] PASS: ${REQUIRED_SUCCESSES} consecutive successes"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo "[healthcheck] FAIL: timed out after ${MAX_WAIT}s without ${REQUIRED_SUCCESSES} consecutive successes"
exit 1
