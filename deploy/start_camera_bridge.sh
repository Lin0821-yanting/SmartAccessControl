#!/usr/bin/env bash
# Copyright (c) 2026 GI104 henrytsai, Yanting Lin
# Tatung University — I4210 AI實務專題
#
# start_camera_bridge.sh — Direction C 相機橋接（host 端）
#
# 在 HOST 上把 CSI（IMX219 / nvarguscamerasrc）擷取到的 BGR 影格推進 POSIX
# 共享記憶體（shmsink）。容器以 --ipc=host + 掛載 socket 目錄後，用 shmsrc
# 讀取（見 src/pipeline/main.py 的 GstShmCapture 與 deploy/docker-compose.yml）。
#
# 為什麼需要這個：容器內的 Argus 無法初始化 EGLDisplay，開不了 CSI 相機；
# 由 host 原生擷取再經共享記憶體餵給容器，是最易複現的方案。
#
# 用法：
#   ./deploy/start_camera_bridge.sh            # 前景執行（Ctrl+C 結束）
#   ./deploy/start_camera_bridge.sh &          # 背景執行
#
# 解析度/路徑可用環境變數覆蓋，需與容器端一致：
#   CAMERA_SHM_SOCKET (預設 /tmp/camshm/cam)
#   CAMERA_SHM_WIDTH  (預設 1280)
#   CAMERA_SHM_HEIGHT (預設 720)
#   CAMERA_SHM_FPS    (預設 30)
#   CAMERA_SENSOR_ID  (預設 0)
#   CAMERA_FLIP       (預設 0；nvvidconv flip-method)
set -euo pipefail

SOCKET="${CAMERA_SHM_SOCKET:-/tmp/camshm/cam}"
WIDTH="${CAMERA_SHM_WIDTH:-1280}"
HEIGHT="${CAMERA_SHM_HEIGHT:-720}"
FPS="${CAMERA_SHM_FPS:-30}"
SENSOR="${CAMERA_SENSOR_ID:-0}"
FLIP="${CAMERA_FLIP:-0}"

# shm-size：容納數張 BGR 影格的環形緩衝（W*H*3 * ~8）。
FRAME_BYTES=$(( WIDTH * HEIGHT * 3 ))
SHM_SIZE=$(( FRAME_BYTES * 8 ))

SOCK_DIR="$(dirname "$SOCKET")"
mkdir -p "$SOCK_DIR"
# 清掉上次殘留的 socket，避免 shmsink 啟動衝突。
rm -f "$SOCKET"

echo "[bridge] CSI sensor-id=$SENSOR  ${WIDTH}x${HEIGHT}@${FPS}  flip=$FLIP"
echo "[bridge] shmsink socket=$SOCKET  shm-size=$SHM_SIZE bytes (~$((SHM_SIZE/1024/1024))MB)"
echo "[bridge] 按 Ctrl+C 結束。"

# 結束時清理殘留的 socket 與 /dev/shm 物件。
cleanup() {
    echo "[bridge] 清理中..."
    rm -f "$SOCKET"
    rm -f /dev/shm/shmpipe.* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec gst-launch-1.0 -e \
    nvarguscamerasrc sensor-id="$SENSOR" ! \
    "video/x-raw(memory:NVMM),width=${WIDTH},height=${HEIGHT},format=NV12,framerate=${FPS}/1" ! \
    nvvidconv flip-method="$FLIP" ! "video/x-raw,format=BGRx" ! \
    videoconvert ! "video/x-raw,format=BGR" ! \
    shmsink socket-path="$SOCKET" shm-size="$SHM_SIZE" \
            wait-for-connection=false sync=false
