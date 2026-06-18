#!/usr/bin/env bash
# Copyright (c) 2026 Yanting Lin, henrytsai
# Tatung University — I4210 AI實務專題
# deploy/deploy.sh — pull new image, restart compose stack, verify health
#
# Usage:
#   bash deploy/deploy.sh <image-tag>
#   bash deploy/deploy.sh v1.0.0
#   bash deploy/deploy.sh sha-abc1234
#
# Called by deploy.yml on the Jetson self-hosted runner after a semver tag push.
# Maintains /var/lib/smartaccess/deployed.txt (current) and
#            /var/lib/smartaccess/deployed.txt.history (audit log).

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
NEW_TAG=${1:?"Usage: $0 <image-tag>"}
REGISTRY=ghcr.io
IMAGE_REPO=lin0821-yanting/smartaccesscontrol
FULL_IMAGE="${REGISTRY}/${IMAGE_REPO}:${NEW_TAG}"

# ── State directory ───────────────────────────────────────────────────────────
STATE_DIR=/var/lib/smartaccess
STATE_FILE="${STATE_DIR}/deployed.txt"
HISTORY_FILE="${STATE_DIR}/deployed.txt.history"
COMPOSE_DIR="$(cd "$(dirname "$0")" && pwd)"   # deploy/ directory

mkdir -p "$STATE_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[deploy] $*"; }
die() { echo "[deploy] FATAL: $*" >&2; exit 1; }

log "====== Smart Access Control deploy: ${NEW_TAG} ======"
log "Image  : ${FULL_IMAGE}"
log "Compose: ${COMPOSE_DIR}/docker-compose.yml"

# ── Save previous tag for rollback ────────────────────────────────────────────
PREV_TAG=$(cat "$STATE_FILE" 2>/dev/null || echo "none")
log "Previous tag: ${PREV_TAG}"

# ── Pull new image ────────────────────────────────────────────────────────────
log "Pulling ${FULL_IMAGE}..."
# || true: tolerate transient auth expiry; uses locally cached image as fallback
IMAGE_TAG="$NEW_TAG" docker compose \
    -f "${COMPOSE_DIR}/docker-compose.yml" \
    pull access-control || true

# ── Stop existing container ────────────────────────────────────────────────────
log "Stopping existing stack..."
IMAGE_TAG="$PREV_TAG" docker compose \
    -f "${COMPOSE_DIR}/docker-compose.yml" \
    down --remove-orphans 2>/dev/null || true

# ── Start new container ───────────────────────────────────────────────────────
log "Starting stack with tag=${NEW_TAG}..."
export IMAGE_TAG="$NEW_TAG"
docker compose \
    -f "${COMPOSE_DIR}/docker-compose.yml" \
    up -d

# ── Health check ─────────────────────────────────────────────────────────────
log "Running health check (up to 120s for TRT engine compilation on first run)..."
if bash "${COMPOSE_DIR}/healthcheck.sh" 120; then
    log "Health check passed."
else
    log "Health check FAILED — initiating automatic rollback to ${PREV_TAG}..."
    bash "${COMPOSE_DIR}/rollback.sh" "$PREV_TAG" || {
        die "Rollback also failed. Manual intervention required."
    }
    die "Deployment failed; rolled back to ${PREV_TAG}."
fi

# ── Update state file ─────────────────────────────────────────────────────────
echo "$NEW_TAG" > "$STATE_FILE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  deployed  ${NEW_TAG}  (was: ${PREV_TAG})" \
    >> "$HISTORY_FILE"

log "====== Deploy complete: ${NEW_TAG} ======"
