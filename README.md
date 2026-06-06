

# Smart Laboratory Access Control System（智慧門禁系統）

> Copyright (c) 2026 GI104 henrytsai — Tatung University 14210 AI實務專題

Edge AI 門禁系統，部署於 **Jetson Orin Nano**，結合超音波距離門控與即時人臉辨識，取代傳統 RFID 感應卡。

## System Overview

```
IMX219 Camera
     │
     ▼
YOLOv8n-face (TensorRT FP16)   ← 人臉偵測
     │
     ├──► MobileFaceNet (ONNX CUDA)  ← 特徵萃取 + Cosine Similarity
     │
     └──► MiniFASNet (ONNX CUDA)     ← 活體偵測
                │
                ▼
        三條件決策引擎
        ├─ similarity ≥ 0.75
        ├─ liveness ≥ 0.6
        └─ 連續 2 幀匹配
                │
                ▼
        MQTT Publish → Kit#2
        GPIO → Servo / LED / Buzzer（Member B）
```

## Hardware

| 元件 | 規格 | 用途 |
|------|------|------|
| Jetson Orin Nano | 8GB | 主運算平台 |
| IMX219 | CSI Camera | 影像輸入 |
| HC-SR04 | 超音波感測器 | 距離門控 |
| SG90 Servo | GPIO Pin 33 | 門鎖控制 |
| Green LED | GPIO Pin 16 | 授權指示 |
| Red LED | GPIO Pin 18 | 拒絕指示 |
| Piezo Buzzer | GPIO Pin 22 | 音效提示 |

## AI Models

| 模型 | 格式 | 精度 | 推論速度 | 用途 |
|------|------|------|----------|------|
| YOLOv8n-face | TensorRT FP16 | - | ~13.6ms | 人臉偵測 + 5 keypoints |
| MobileFaceNet | ONNX CUDA | FP32 | ~4.5ms | 128-dim embedding |
| MiniFASNet | ONNX CUDA | FP32 | ~4.9ms | 活體偵測 |
| **Total** | | | **~25.5ms** | **目標 < 500ms ✅** |

## Project Structure

```
SmartAccessControl/
├── configs/
│   └── config.yaml          # 所有可調參數
├── data/
│   ├── enrollment/          # 人臉照片（每人資料夾）
│   └── face_db.npy          # 特徵資料庫（gitignored）
├── models/
│   ├── weights/             # ONNX 權重
│   └── engines/             # TensorRT engine
├── scripts/
│   ├── collect_faces.py     # 收集 enrollment 照片
│   └── enroll.py            # 建立 face_db
├── src/
│   ├── detection/
│   │   └── detector.py      # YOLOv8n-face 推論器
│   ├── recognition/
│   │   └── recognizer.py    # MobileFaceNet + cosine similarity
│   ├── antispoof/
│   │   └── antispoof.py     # MiniFASNet 活體偵測
│   ├── gpio/
│   │   └── gpio.py          # Servo / LED / Buzzer（Member B）
│   ├── mqtt/
│   │   └── publisher.py     # MQTT 事件發布
│   └── pipeline/
│       └── main.py          # 主程式
└── tests/
    ├── unit/
    └── integration/
```

## Setup

### 1. 環境需求

- JetPack 6.x（CUDA 12.6、TensorRT 10.7、OpenCV 4.10）
- Python 3.10
- PDM 2.x

### 2. 安裝依賴

```bash
git clone https://github.com/Lin0821-yanting/SmartAccessControl.git
cd SmartAccessControl
pip install pdm
pdm install
```

### 3. 系統套件軟連結（Jetson 必要步驟）

```bash
VENV_SITE="/home/jetson/SmartAccessControl/.venv/lib/python3.10/site-packages"
SYS1="/usr/local/lib/python3.10/dist-packages"
SYS2="/usr/lib/python3/dist-packages"
LOCAL="/home/jetson/.local/lib/python3.10/site-packages"

for pkg in mpmath sympy contourpy dateutil fonttools pandas scipy seaborn tqdm requests \
           ultralytics ultralytics_thop torch torchvision torchgen PIL numpy numpy.libs \
           matplotlib mpl_toolkits charset_normalizer onnx onnxslim onnxruntime \
           omegaconf antlr4 google; do
    [ -e $SYS1/$pkg ] && ln -sf $SYS1/$pkg $VENV_SITE/$pkg
done
for pkg in urllib3 idna certifi; do
    [ -e $SYS2/$pkg ] && ln -sf $SYS2/$pkg $VENV_SITE/$pkg
done
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
# 釋放記憶體後執行
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

pdm run python -c "
from ultralytics import YOLO
model = YOLO('models/weights/yolov8n-face.pt')
model.export(format='engine', imgsz=416, half=True, device=0)
"
mv models/weights/yolov8n-face.engine models/engines/yolov8n-face-fp16.engine
```

### 6. Enrollment（註冊授權人員）

```bash
# 收集人臉照片（30 張）
pdm run python scripts/collect_faces.py --name <your_name> --count 30

# 建立特徵資料庫
pdm run python scripts/enroll.py
```

### 7. 啟動門禁系統

```bash
# 啟動 MQTT broker
sudo systemctl start mosquitto

# 啟動門禁 pipeline
pdm run python src/pipeline/main.py

# Headless 模式（無螢幕）
pdm run python src/pipeline/main.py --no-display
```

## Decision Logic

系統採用**三條件同時滿足**才授權：

```
1. Cosine Similarity ≥ 0.75   （與 face_db 的相似度）
2. Liveness Score   ≥ 0.6    （MiniFASNet 真人機率）
3. 連續 2 幀匹配同一人          （時間一致性，防單幀照片攻擊）
```

## MQTT Topics

| Topic | 方向 | Payload |
|-------|------|---------|
| `lab/access/events` | Kit#1 → Kit#2 | identity, similarity, liveness, granted, reason, timestamp |
| `lab/access/status` | Kit#1 → Kit#2 | door_state, last_person, timestamp |
| `lab/access/heartbeat` | Kit#1 → Kit#2 | fps, distance_cm, timestamp |

### Event Payload 範例

```json
{
  "identity": "henry",
  "similarity": 0.8904,
  "liveness": 0.9938,
  "granted": true,
  "reason": "similarity=0.890, liveness=0.994",
  "timestamp": "2026-06-06T15:13:40"
}
```

## Known Limitations

- MiniFASNet 對高品質印刷照片或高解析度螢幕仍有誤判風險（Week 14 測試記錄）
- Enrollment 照片需在相似光線環境下拍攝，否則 similarity 會下降
- 單人場景設計，多人同框時只處理信心度最高的人臉

## Member Contributions

| Member | 負責範圍 |
|--------|----------|
| Member A（henrytsai） | 模型訓練、TensorRT 匯出、AI pipeline、MQTT publisher |
| Member B | GPIO 控制、HC-SR04 距離門控、Kit#2 MQTT subscriber、系統整合 |