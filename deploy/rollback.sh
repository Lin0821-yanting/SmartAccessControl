#!/usr/bin/env bash
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
# deploy/rollback.sh — revert to previous deployed tag in < 30 s
#
# Usage:
#   bash deploy/rollback.sh             # reads previous tag from state file
#   bash deploy/rollback.sh <tag>       # roll back to a specific tag
#
# Records timing: `time bash deploy/rollback.sh` for the README evidence.

set -euo pipefail

STATE_DIR=/var/lib/smartaccess
STATE_FILE="${STATE_DIR}/deployed.txt"
HISTORY_FILE="${STATE_DIR}/deployed.txt.history"
COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"

REGISTRY=ghcr.io
IMAGE_REPO=lin0821-yanting/smartaccesscontrol

log() { echo "[rollback] $*"; }
die() { echo "[rollback] FATAL: $*" >&2; exit 1; }

# ── Determine target tag ──────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    ROLLBACK_TAG="$1"
    log "Rolling back to explicitly specified tag: ${ROLLBACK_TAG}"
else
    # Read the second-to-last line of history (previous deployment)
    ROLLBACK_TAG=$(grep ' deployed ' "$HISTORY_FILE" 2>/dev/null \
        | tail -2 | head -1 \
        | awk '{print $3}' \
        || echo "")

    if [ -z "$ROLLBACK_TAG" ] || [ "$ROLLBACK_TAG" = "none" ]; then
        die "No previous tag found in ${HISTORY_FILE}. Cannot roll back."
    fi
    log "Rolling back to previous tag from history: ${ROLLBACK_TAG}"
fi

FULL_IMAGE="${REGISTRY}/${IMAGE_REPO}:${ROLLBACK_TAG}"
CURRENT_TAG=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")
log "Current tag  : ${CURRENT_TAG}"
log "Rollback tag : ${ROLLBACK_TAG}"

# ── Pull (no-op if already cached — speeds up rollback) ───────────────────────
log "Pulling ${FULL_IMAGE} (no-op if cached)..."
IMAGE_TAG="$ROLLBACK_TAG" docker compose \
    -f "${COMPOSE_DIR}/docker-compose.yml" \
    pull access-control || true   # tolerate auth expiry; use cached image

# ── Stop current container ─────────────────────────────────────────────────────
log "Stopping current stack..."
IMAGE_TAG="$CURRENT_TAG" docker compose \
    -f "${COMPOSE_DIR}/docker-compose.yml" \
    down --remove-orphans 2>/dev/null || true

# ── Start rollback container ───────────────────────────────────────────────────
log "Starting rollback stack with tag=${ROLLBACK_TAG}..."
export IMAGE_TAG="$ROLLBACK_TAG"
docker compose \
    -f "${COMPOSE_DIR}/docker-compose.yml" \
    up -d

# ── Verify rollback is healthy ─────────────────────────────────────────────────
log "Verifying rollback health..."
if bash "${COMPOSE_DIR}/healthcheck.sh" 60; then
    log "Rollback health check passed."
else
    die "Rollback to ${ROLLBACK_TAG} also failed — both tags are broken. Alert the team."
fi

# ── Update state file ──────────────────────────────────────────────────────────
echo "$ROLLBACK_TAG" > "$STATE_FILE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  rollback  ${ROLLBACK_TAG}  (was: ${CURRENT_TAG})" \
    >> "$HISTORY_FILE"

log "====== Rollback complete: now running ${ROLLBACK_TAG} ======"
