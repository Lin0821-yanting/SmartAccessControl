# Smart Laboratory Access Control System（智慧門禁系統）

> Copyright (c) 2026 GI104 henrytsai, Yanting Lin — Tatung University I4210 AI實務專題

Edge AI 門禁系統，部署於 **Jetson Orin Nano**，結合超音波距離門控與即時人臉辨識 + 活體偵測，取代傳統 RFID 感應卡。

## System Overview

```
IMX219 Camera
     │
     ▼
YOLOv8n-face (TensorRT FP16)   ← 人臉偵測
     │
     ├──► MobileFaceNet (ONNX CUDA)  ← 特徵萃取 + Cosine Similarity
     │
     └──► MiniFASNet (ONNX CUDA)     ← 活體偵測（吃整張畫面）
                │
                ▼
        決策引擎（問題 2 調校後）
        ├─ similarity ≥ 0.5
        ├─ liveness   ≥ 0.3
        ├─ GRANT  需連續 4 幀匹配同一人
        └─ SPOOF  需連續 5 幀活體失敗
                │
                ▼
        MQTT Publish → Kit#2
        GPIO → Servo / LED / Buzzer
```

## Hardware

| 元件 | 規格 | GPIO (BOARD) | 用途 |
|------|------|------|------|
| Jetson Orin Nano | 8GB | — | 主運算平台 |
| IMX219 | CSI Camera | CSI-0 | 影像輸入 |
| HC-SR04 | 超音波感測器 | TRIG 31 / ECHO 15 | 距離門控 |
| SG90 Servo | 門鎖 | 33 | 門鎖控制 |
| Green LED | 授權指示 | 7 | GRANT |
| Red LED | 拒絕指示 | 11 | DENY / UNKNOWN / SPOOF |
| Piezo Buzzer | 音效 | 29 | SPOOF 警報 |

> 腳位以 `configs/config.yaml` 的 `gpio:` 區段為準。

## AI Models

| 模型 | 格式 | 推論速度 | 用途 |
|------|------|----------|------|
| YOLOv8n-face | TensorRT FP16 | ~13.6 ms | 人臉偵測 + 5 keypoints |
| MobileFaceNet | ONNX CUDA | ~4.5 ms | 128-dim embedding |
| MiniFASNet | ONNX CUDA | ~4.9 ms | 活體偵測 |
| **AI 推論合計** | | **~25.5 ms** | Decide+Act <2 ms、MQTT <3 ms |

> 系統的端到端速率受 **相機幀率（~20 FPS）** 限制，而非算力——推論僅佔每幀預算的一小部分（見 Reflection）。

## Project Structure

```
SmartAccessControl/
├── configs/config.yaml          # 所有可調參數（GPIO / 門檻 / MQTT）
├── data/
│   ├── enrollment/<name>/        # 每人 enrollment 照片
│   └── face_db.npy               # 特徵資料庫
├── models/{weights,engines}/     # ONNX 權重 / TensorRT engine
├── scripts/
│   ├── collect_faces.py          # 收集 enrollment 照片
│   ├── enroll.py                 # 建立 face_db
│   ├── record_session.py         # 情境錄製（倒數緩衝 + 標記）
│   └── analyze_decisions.py      # 依標記統計決策 log
├── src/
│   ├── detection/detector.py     # YOLOv8n-face 推論器
│   ├── recognition/recognizer.py # MobileFaceNet + cosine similarity
│   ├── antispoof/antispoof.py    # MiniFASNet 活體偵測
│   ├── pipeline/main.py          # 主程式（Sense→Decide→Act→Publish）
│   ├── decision_engine.py        # 純邏輯決策狀態機
│   ├── actuator_controller.py    # Servo / LED / Buzzer 門面
│   ├── led.py / buzzer.py / servo.py / hc_sr04.py
│   ├── mqtt_publisher.py         # MQTT 事件發布
│   └── orchestrator.py           # 整合迴圈（保留）
├── tests/                        # 單元測試 + 覆蓋率/accuracy gate
│   └── integration/              # Jetson 整合測試
├── deploy/                       # docker-compose / deploy.sh / 相機橋接
├── accuracy_baseline.json        # accuracy gate 基準
└── .github/workflows/            # ci.yml（5 階段）/ deploy.yml
```

---

# Quickstart

## Setup

### 1. 環境需求
- JetPack 6.x（L4T R36，CUDA 12.6、TensorRT 10.7、OpenCV 4.10）
- Python 3.10、PDM 2.x

### 2. 安裝依賴
```bash
git clone https://github.com/Lin0821-yanting/SmartAccessControl.git
cd SmartAccessControl
pip install pdm && pdm install
```

### 3. 系統套件軟連結（Jetson 必要步驟）
Jetson 的 CUDA/Torch/TensorRT 系統套件需軟連結進 venv：
```bash
VENV_SITE="$PWD/.venv/lib/python3.10/site-packages"
SYS1="/usr/local/lib/python3.10/dist-packages"
SYS2="/usr/lib/python3/dist-packages"
LOCAL="/home/jetson/.local/lib/python3.10/site-packages"

for pkg in mpmath sympy contourpy dateutil fonttools pandas scipy seaborn tqdm requests \
           ultralytics ultralytics_thop torch torchvision torchgen PIL numpy numpy.libs \
           matplotlib mpl_toolkits charset_normalizer onnx onnxslim onnxruntime \
           omegaconf antlr4 google; do
    [ -e $SYS1/$pkg ] && ln -sf $SYS1/$pkg $VENV_SITE/$pkg
done
for pkg in urllib3 idna certifi; do [ -e $SYS2/$pkg ] && ln -sf $SYS2/$pkg $VENV_SITE/$pkg; done
ln -sf $SYS2/six.py $VENV_SITE/six.py
ln -sf $SYS2/pyparsing.py $VENV_SITE/pyparsing.py
for pkg in certifi cycler kiwisolver omegaconf antlr4; do
    [ -e $LOCAL/$pkg ] && ln -sf $LOCAL/$pkg $VENV_SITE/$pkg
done
```

### 4. 下載模型權重
| 模型 | 來源 | 存放路徑 |
|------|------|----------|
| YOLOv8n-face | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) | `models/weights/yolov8n-face.pt` |
| MobileFaceNet | [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet) | `models/weights/MobileFaceNet.onnx` |
| MiniFASNet | [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) | `models/weights/minifasnet.onnx` |

### 5. 匯出 TensorRT Engine
```bash
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'   # 先釋放記憶體
pdm run python -c "from ultralytics import YOLO; \
  YOLO('models/weights/yolov8n-face.pt').export(format='engine', imgsz=416, half=True, device=0)"
mv models/weights/yolov8n-face.engine models/engines/yolov8n-face-fp16.engine
```

## 建立 face_db（Enrollment）

```bash
# (a) 收集人臉照片（每人約 30 張，存到 data/enrollment/<name>/）
pdm run python scripts/collect_faces.py --name <name> --count 30

# (b) 萃取特徵、L2 正規化取平均，寫入 data/face_db.npy
pdm run python scripts/enroll.py --name <name>
```

## MQTT Topic Map

| Topic | 方向 | 主要欄位 |
|-------|------|----------|
| `lab/access/events` | Kit#1 → Kit#2 | `decision`(GRANT/DENY/UNKNOWN/SPOOF/IGNORE), `identity`, `similarity`, `spoof_score`, `is_live`, `face_in_db`, `consecutive_frames`, `bbox` |
| `lab/access/status` | Kit#1 → Kit#2 | `door_state`(locked/unlocked), `last_person` |
| `lab/access/heartbeat` | Kit#1 → Kit#2 | `fps`, `cpu_temp_c`, `ram_used_gb`, `distance_cm`, `pipeline_stage`, `container_uptime_s` |

Broker 由 `deploy/docker-compose.yml` 的 `mosquitto` service 提供（或 host 上 `sudo systemctl start mosquitto`）。

## How to Run

### A. Host 原生（最單純，相機直接走 CSI）
```bash
sudo systemctl start mosquitto
pdm run python src/pipeline/main.py              # 有畫面
pdm run python src/pipeline/main.py --no-display  # headless
```

### B. Docker（容器跑完整 AI pipeline，相機走共享記憶體橋接）
```bash
# 拉 image（或本機已 build 的 sha- 標籤）
docker pull ghcr.io/lin0821-yanting/smartaccesscontrol:latest

# host 端啟動 CSI → 共享記憶體橋接
./deploy/start_camera_bridge.sh &

# 起 mosquitto + access-control（自動 --source shm）
cd deploy && IMAGE_TAG=latest docker compose up
```
> 容器內無法直接開 CSI（Argus EGLDisplay 限制），故採共享記憶體橋接——詳見 [Docker 鏡頭解法](#docker-鏡頭解法解決問題-1)。

## How to Demo

用 `record_session.py` 包住主程式，每個情境開始前**倒數緩衝**（給站位/拿照片時間），並把情境起訖標記寫進帶時間戳的 log；事後用 `analyze_decisions.py` 切段統計。

```bash
python scripts/record_session.py -- pdm run python src/pipeline/main.py --no-display
#   輸入情境編號(1=S1…) → 倒數 → 就定位 → 完成按 Enter；q 離開
python scripts/analyze_decisions.py --csv logs/summary.csv   # 產出統計表
```

測試情境（每個隔離一個決策關卡，B=已註冊、C=臨時未註冊）：

| 情境 | 對象 | 輸入 | 預期 | 隔離的關卡 |
|---|---|---|---|---|
| S1 | B | 真臉 @100cm | GRANT | recog+活體+連續4幀 |
| S2/S3 | B | 照片 / 螢幕 | SPOOF | 活體擋 2D 平面 |
| S4 | C | 真臉 | UNKNOWN | DB 比對 |
| S5 | C | 照片 | SPOOF | 優先序：活體 > unknown |
| S6 | — | 無人 | IGNORE | HC-SR04 距離 gate |
| S7 | B | 一閃而過 | 不開門 | 連續幀時間一致性 |
| S10 | B | 戴眼鏡/口罩 | 非 SPOOF | 對照組：不無差別判假 |

---

# Architecture

## CI/CD Pipeline（`.github/workflows/ci.yml` + `deploy.yml`）

```
lint ──┬──► test ───────────────┐
       │                        ├──► build ──► integration-test
       └──► security-scan ──────┘                    (main only)
                                          tag push ──► deploy
```

| 階段 | Runner | 內容 | 閘門 |
|------|--------|------|------|
| **Lint** | ubuntu | `ruff check .` + `ruff format --check .` | 零違規 |
| **Test** | ubuntu | `pytest`（推論/訊息單元 + **accuracy gate** 讀 `accuracy_baseline.json`） | **coverage ≥ 90%** |
| **Security-Scan** | ubuntu | `bandit -r src`（SAST）+ `pip-audit`（CVE） | 零高風險 |
| **Build** | ubuntu + QEMU | `docker buildx` ARM64 交叉編譯 → push GHCR `sha-<short>` | 確認 manifest 為 arm64 |
| **Integration-Test** | **self-hosted Jetson** | pull 該 commit image，跑 mock 整合測試（IT-1~6）+ 真實 GPIO smoke（IT-7） | 僅 `main` push 觸發 |
| **Deploy** | **self-hosted Jetson** | 版本 **tag push** 觸發：把 SHA image 重新標 semver（不重 build）→ 跑 `deploy/deploy.sh`（healthcheck + rollback）→ production | tag-triggered |

- Lint / Test / Security 在便宜的 ubuntu runner；Build 用 QEMU 交叉編 ARM64。
- Integration-Test 在 Jetson 自架 runner（`[self-hosted, linux, arm64, jetson]`）跑真實硬體。
- Deploy 是獨立 workflow，只在打版本 tag（如 `v1.0.0`）時把已驗證的 image 上 production，**不重新 build**。

---

# 參數設置（解決問題 2）

> 重點不在模型準確率，而在**如何用系統設計補償一個無法重新訓練的弱活體模型**。

## 問題現象
主程式運行時，真人站在鏡頭前被 MiniFASNet 持續判為 SPOOF → 蜂鳴器對真人不停誤響，且真人**永遠拿不到 GRANT**。

## 根因
三個模型皆為**預訓練、未自行訓練**，只能調「輸入構圖」與「決策邏輯」。單張乾淨註冊照下 MiniFASNet 對真人 `real_prob ≈ 0.95`（模型沒壞），問題在即時鏡頭輸入分布與訓練分布不一致：

| 因素 | 說明 |
|---|---|
| 構圖/距離（主因） | 緊裁切讓臉塞滿畫面、缺背景 → 活體分數崩潰 |
| 光線 | 閃光側打造成高光/硬陰影，被當成螢幕/列印反光 |
| Domain gap | 訓練相機與 IMX219 色調/曝光不同 |

真人 live 中位數隨距離：60cm→0.13、60cm+打光→0.19、**100cm（含背景）→0.25–0.35**。整張畫面餵活體（vs 緊裁切）median 0.19 → **0.76**。

## 最終參數（在「不重新訓練」前提下）

| 參數 | 值 | 理由 |
|---|---|---|
| 活體輸入構圖 | **整張畫面**（`LIVENESS_FRAME=full`） | 對齊註冊照/訓練分布 |
| HC-SR04 gate | **100 cm**（原 60） | 站遠、臉變小、背景進來 |
| `LIVENESS_THRESHOLD` | **0.3**（原 0.6） | 真人 median 0.35 vs 假照片 0.13，分界落在 0.3 |
| GRANT 連續幀 | **4** | 用時間一致性指數放大微弱區分 |
| SPOOF 連續幀 | **5** | 真人偶爾掉分不誤報；照片持續失敗才警報 |
| 警報冷卻 | **5 s** | ~20FPS 下把「每幀觸發」限制為「每 5s 最多一次」 |
| `similarity_threshold` | **0.5** | 與 `config.yaml` 一致 |

**核心論述**：真人 live median 0.261 vs 假照片 0.174，模型只有微弱區分；但「GRANT 連續 4 幀（真人 0.56⁴≈10% vs 照片 0.21⁴≈0.2%）+ SPOOF 連續 5 幀」把差距**指數放大**成可靠決策。

## 驗證結果（held-out，gate=100cm）
| 情境 | live median | 開門 | 蜂鳴器 | 判定 |
|---|---|---|---|---|
| 真人（112 幀） | 0.261 | ✅ 3 次 | **0 次** | 能進、不吵 ✅ |
| 假照片（340 幀） | 0.174 | ❌ **0 次** | ✅ 3 次 | 擋下、會警報 ✅ |

四目標同時達成：①真人開門 ②真人不嗶 ③假照片警報 ④假照片絕不開門。這些數字即 `accuracy_baseline.json` 的 accuracy gate 基準。

無需改碼即可現場微調（環境變數）：`LIVENESS_THRESHOLD`、`SPOOF_FRAMES`、`LIVENESS_FRAME`。

---

# Docker 鏡頭解法（解決問題 1）

## 問題
打包成 image 後在容器內 `nvarguscamerasrc` 開不了 IMX219 CSI 相機。

## 根因（逐層排查）
CSI 不走 `/dev/video0`，而是容器內 `nvarguscamerasrc` → host `nvargus-daemon`（`/tmp/argus_socket`）→ ISP，且整條 Argus/EGL userspace 要齊全：

| 排查層 | 處置 / 結果 |
|---|---|
| Argus socket 未掛 | 掛 `/tmp/argus_socket` → 連上 daemon |
| 容器無 `nvarguscamerasrc` plugin | 掛 `libgstnvarguscamerasrc.so` |
| plugin 缺 `libGLESv2.so.2` | 補掛 GLES |
| tegra libs | 旁路掛 `/usr/lib/.../nvidia` + `tegra-egl` |
| **EGLDisplay 初始化失敗** | 補 `/dev/dri` 仍 `Failed to initialize EGLDisplay` → **JetPack6/R36 硬牆，放棄此路** |

> host R36.4.7、容器 base R36.4.0（版本相容，非版本問題）。

## 解法：共享記憶體橋接（Direction C）
不讓容器碰 Argus；**host 原生擷取 → POSIX 共享記憶體 → 容器讀取**：
```
[host]      nvarguscamerasrc ! nvvidconv ! BGR ! shmsink   ← deploy/start_camera_bridge.sh
                                  │ POSIX shm (/dev/shm)
[container] shmsrc ! appsink → numpy BGR                   ← GstShmCapture（--source shm）
```
- 容器須 `--ipc=host`（POSIX shm 在 `/dev/shm`，需共用 IPC namespace）。
- 容器 cv2 未編 GStreamer → 用 **PyGObject appsink** 讀影格，包成 cv2 相容介面，主迴圈不變。
- 容器完全不需 Argus/EGL/nvidia 相機 lib；shm 緩衝僅 ~20MB（非 GB）。

| 方案 | 容器跑 AI | 複現難度 | 結論 |
|---|---|---|---|
| A 相機進容器 | — | EGLDisplay 硬牆 | ❌ |
| B pipeline 跑 host | ❌ | 需 host 全套依賴 | 可行但非容器化 |
| **C shm 橋接（採用）** | ✅ | `compose up` + 一行 host 指令 | ✅ |

---

# Reflection

- **系統是 camera-rate 受限，而非算力受限**：AI 推論僅 ~25.5 ms/幀，Decide+Act <2 ms、MQTT <3 ms，但相機僅 ~20 FPS，端到端瓶頸在感測輸入而非運算。事前未明確設定效能目標，回溯來看延遲遠低於門禁可接受範圍（人走到門前的反應時間），因此最佳化重心從「加速」轉為「**用系統設計補償弱模型**」。
- **最大的工程教訓是「不要硬幹底層」**：問題 1 我們逐層補 socket→plugin→GLES→tegra libs→/dev/dri，最後仍卡 EGLDisplay；認清這是平台硬牆後改走共享記憶體橋接，幾小時內解決。判斷「何時停止鑽牛角尖、換架構」比繼續補洞更重要。
- **弱模型用時間一致性救回**：MiniFASNet 在即時鏡頭逐幀很吵，單一門檻無法乾淨分離真人/照片；靠「連續 N 幀」把微弱的逐幀差距指數放大，配合警報冷卻，才同時達成「真人能進、真人不吵、照片擋下、照片不開門」。
- **誠實的限制**：真人與照片 live 分數仍重疊，極端單幀仍可能誤判；受測 live 對象僅 B + C 屬小樣本，重點放在**測試情境設計的完整性**（涵蓋每個決策關卡 + 對照組）而非統計顯著性。

## Known Limitations
- MiniFASNet 對高品質印刷照/高解析螢幕仍有誤判風險；靠站遠 + 整張畫面 + 連續幀補償。
- Enrollment 照片需在相似光線下拍攝，否則 similarity 下降。
- 單人場景設計，多人同框只處理信心度最高的人臉。

## Member Contributions
| Member | 負責範圍 |
|--------|----------|
| henrytsai（GI104） | 模型訓練/TensorRT 匯出、AI pipeline、活體與決策調校、MQTT publisher |
| Yanting Lin | GPIO 控制、HC-SR04 距離門控、Docker/CI-CD、共享記憶體相機橋接、系統整合 |
