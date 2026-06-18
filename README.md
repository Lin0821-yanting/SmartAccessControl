
## Jetson 環境軟連結設定

由於 Jetson 系統套件需透過軟連結引入 venv，執行以下指令：

```bash
VENV_SITE="/home/jetson/SmartAccessControl/.venv/lib/python3.10/site-packages"
SYS1="/usr/local/lib/python3.10/dist-packages"
SYS2="/usr/lib/python3/dist-packages"
LOCAL="/home/jetson/.local/lib/python3.10/site-packages"

for pkg in mpmath sympy contourpy dateutil fonttools pandas scipy seaborn tqdm requests ultralytics ultralytics_thop torch torchvision torchgen PIL numpy numpy.libs matplotlib mpl_toolkits charset_normalizer; do
    [ -e $SYS1/$pkg ] && ln -sf $SYS1/$pkg $VENV_SITE/$pkg
done
for pkg in urllib3 idna certifi; do
    [ -e $SYS2/$pkg ] && ln -sf $SYS2/$pkg $VENV_SITE/$pkg
done
ln -sf $SYS2/six.py $VENV_SITE/six.py
ln -sf $SYS2/pyparsing.py $VENV_SITE/pyparsing.py
for pkg in certifi cycler kiwisolver; do
    [ -e $LOCAL/$pkg ] && ln -sf $LOCAL/$pkg $VENV_SITE/$pkg
done
```
#MODELS
```
YOLOFACE
https://github.com/Yusepp/YOLOv8-Face.git

MOBILEFACENET
https://github.com/foamliu/MobileFaceNet.git

MINIFASNET
https://github.com/facenox/face-antispoof-onnx.git
```

---

## 活體偵測與蜂鳴器設計決策（SPOOF / Buzzer Design Rationale）

> 本節記錄期末階段針對「活體模型在即時鏡頭下表現不穩，導致蜂鳴器對真人持續誤觸發」所做的問題分析、設計決策與實測結果。重點不在模型準確率，而在**如何用系統設計補償一個無法重新訓練的弱模型**。

### 1. 問題現象

主程式運行時，真人站在鏡頭前會被活體模型（MiniFASNet）持續判為 SPOOF，導致：

- 蜂鳴器每隔數秒就響一次，幾乎停不下來。
- 真人因一直被判 SPOOF，**永遠拿不到 GRANT**，門禁無法正常授權。

### 2. 根因分析

三個模型（YOLOv8n-face / MobileFaceNet / MiniFASNet）皆為**預訓練模型，未經我們自行訓練**，因此只能調整「餵進模型的輸入」與「決策邏輯」，無法改權重。

以單張乾淨註冊照測試，MiniFASNet 對真人可正確判出 `real_prob ≈ 0.95`，**模型本身沒有壞**。問題出在即時鏡頭的輸入分布與訓練分布不一致：

| 因素 | 說明 |
|---|---|
| **構圖／距離（主因）** | MiniFASNet 靠臉周圍的背景與深度線索判真假。偵測器的緊裁切讓臉塞滿畫面、缺乏背景 → 分數崩潰。 |
| **光線** | 手機閃光側打造成高光與硬陰影，被模型視為螢幕/列印反光。 |
| **Domain gap** | 訓練資料相機與 IMX219 CSI 相機色調/曝光不同。 |

關鍵量測（真人 live 分數中位數隨距離變化）：

| 條件 | live median |
|---|---|
| 60cm（緊距離） | 0.13 |
| 60cm + 打光 | 0.19 |
| **100cm（站遠、含背景）** | **0.25 – 0.35** |

註冊照本身為 1920×1080 整張畫面（臉小、背景多），餵整張畫面給活體模型的分數遠高於緊裁切：

| 餵入活體模型的構圖 | live median（239 張註冊照） |
|---|---|
| **整張畫面** | **0.765** |
| 緊裁切（原始做法） | ~0.19 |
| 中央方形 | 0.441 |

### 3. 設計決策（在「不重新訓練」前提下）

| # | 決策 | 理由 |
|---|---|---|
| 1 | **活體模型改吃整張畫面**（非緊裁切） | 對齊註冊照/訓練分布，真人 median 0.19 → 0.76（離線）。 |
| 2 | **HC-SR04 gate 放寬 60cm → 100cm** | 讓使用者站遠、臉變小、背景進來，即時 live median 拉到 0.25–0.35。 |
| 3 | **活體門檻 0.6 → 0.3** | 真人 median 0.35 vs 假照片 median 0.13，分界點落在 0.3 附近。 |
| 4 | **GRANT 需連續 4 幀、SPOOF 需連續 5 幀**（時間一致性） | 見下方「核心設計論述」。 |
| 5 | **警報冷卻 5 秒**（debounce） | 相機 ~20 FPS，若每幀觸發蜂鳴器會連續狂叫；冷卻將「每幀一次」限制為「每 5 秒最多一次」。 |

#### 核心設計論述：用時間一致性放大微弱的區分

實測真人與假照片的 live 分數**重疊嚴重**，單一門檻無法乾淨分離：

- 門檻太低 → 真人能過，但假照片也會偶爾過（demo 時照片開門＝災難）。
- 門檻太高 → 假照片擋得住，但真人過不了、又開始嗶。

但真人的**逐幀**過關率仍高於假照片。利用「**連續 N 幀**」會把這個差距**指數放大**（門檻 0.3 下）：

| 連續要求 | 真人（過關率 ~56%） | 假照片（過關率 ~21%） |
|---|---|---|
| GRANT 連續 4 幀都過 | 0.56⁴ ≈ **10%** → 約 1 秒內開門 | 0.21⁴ ≈ **0.2%** → 幾乎不可能誤開門 |
| SPOOF 連續 5 幀都失敗 | 0.44⁵ ≈ **1.6%** → 幾乎不嗶 | 0.79⁵ ≈ **31%** → 持續嗶、警報成立 |

> **一句話總結**：真人 live median 0.261 vs 假照片 0.174，模型本身只有微弱區分，但靠「GRANT 連續 4 幀 + SPOOF 連續 5 幀」把它放大成可靠決策。

### 4. 測試方法

- 在 gate=100cm、相同距離下，分別錄製「真人」與「手機照片」各一段，從決策 log 擷取每幀 `live` 分數。
- 比較兩條分布，於重疊區間挑選分界門檻（0.3）。
- 以最終參數重跑雙段驗證，統計四個關鍵事件：真人是否開門 / 真人是否誤觸蜂鳴器 / 假照片是否觸發蜂鳴器 / 假照片是否誤開門。

### 5. 最終驗證結果

最終參數：`gate=100cm`、活體吃整張畫面、`LIVENESS_THRESHOLD=0.3`、`confirm_frames(GRANT)=4`、`SPOOF_FRAMES=5`、警報冷卻 5s。

| 情境 | live median | 開門 (GRANT) | 蜂鳴器 | 判定 |
|---|---|---|---|---|
| **真人**（112 幀） | 0.261 | ✅ 3 次（綠燈） | **0 次（靜音）** | 真人能進、不吵 ✅ |
| **假照片**（340 幀） | 0.174 | ❌ **0 次（門未開）** | ✅ 3 次（每 5s 一組） | 照片擋下、會警報 ✅ |

四個目標同時達成：① 真人開門　② 真人不嗶　③ 假照片警報　④ 假照片絕不開門。

### 6. 可調參數（環境變數）

無需改碼即可現場微調：

```bash
LIVENESS_THRESHOLD=0.3   # 活體通過門檻（真人/假照片分界）
SPOOF_FRAMES=5           # 連續幾幀判假才算 SPOOF（調高=真人更安靜）
LIVENESS_FRAME=full      # 活體輸入構圖：full（整張）/ crop（擴大臉框）
```

例：demo 想更安靜 → `SPOOF_FRAMES=6 pdm run python src/pipeline/main.py --no-display`

### 7. 已知限制

- 活體模型在近距離（<60cm）對真人表現不佳；本系統靠「站遠 + 整張畫面 + 時間一致性」補償，而非模型本身可靠。
- 真人與假照片的分數重疊仍在，極端情況下仍可能單幀誤判；連續幀機制將其影響降到可接受範圍。
- demo 操作建議：站約 100cm、臉清楚入鏡、正面均勻光（避免閃光側打）。

---

## 容器化部署：CSI 相機跨容器橋接（Camera-in-Container / Direction C）

> 本節記錄「將主程式打包成 image 後，Docker 容器內讀不到 IMX219 CSI 相機、無法使用 GStreamer」的問題分析與最終解法。重點不在硬塞 driver，而在**用最易複現的架構繞過容器內 Argus 的限制**。

### 1. 問題現象

主程式在 host 上可正常開相機，但打包進 Docker 後，容器內 `nvarguscamerasrc` 無法擷取影像，整條 CSI/GStreamer pipeline 失效。

### 2. 根因分析（逐層排查）

CSI 相機不走 `/dev/video0`（那是 V4L2 介面），而是：容器內 `nvarguscamerasrc` → host 的 `nvargus-daemon`（透過 Unix socket `/tmp/argus_socket`）→ ISP，且整條 Argus/EGL userspace 必須齊全。逐層補洞的實測結果：

| 排查層 | 現象 | 處置 |
|---|---|---|
| Argus socket | 容器未掛 `/tmp/argus_socket` | 掛入 → 成功連上 daemon |
| GStreamer plugin | 容器無 `nvarguscamerasrc`（base image 未含） | 掛入 `libgstnvarguscamerasrc.so` |
| plugin 依賴 | `libGLESv2.so.2: not found`，plugin 載不起來 | 補掛 GLES |
| tegra libs | 需一整包 `/usr/lib/.../nvidia` + `tegra-egl` | 旁路掛載 + `LD_LIBRARY_PATH` |
| **EGLDisplay** | plugin 載入、Argus 連上 daemon，但 **`Failed to initialize EGLDisplay`**（FrameConsumer 建不起來），補 `/dev/dri` 仍失敗 | **JetPack 6 / L4T R36 已知硬牆，放棄此路** |

> host L4T = R36.4.7、容器 base = R36.4.0（同 major.minor，**非版本不相容**）。問題在容器內 Argus 無法初始化 EGLDisplay，再往下需動到顯示／driver 層，兩天死線內成本過高。

### 3. 解決方案：共享記憶體橋接（Direction C）

不讓容器碰 Argus；改由 **host 原生擷取 → POSIX 共享記憶體 → 容器讀取**：

```
[host]      nvarguscamerasrc ! nvvidconv ! BGR ! shmsink     ← deploy/start_camera_bridge.sh
                                  │  POSIX 共享記憶體 (/dev/shm)
[container] shmsrc ! appsink → numpy BGR                     ← GstShmCapture（--source shm）
```

設計要點：

- **容器須 `--ipc=host`**：shmsink/shmsrc 透過 POSIX 共享記憶體（`/dev/shm/shmpipe.*`）傳影格，容器要與 host 共用 IPC namespace 才看得到該物件。
- **容器 cv2 未編 GStreamer**（`getBuildInformation` 顯示 GStreamer: NO，只有 FFMPEG），故不能用 `cv2.VideoCapture(...CAP_GSTREAMER)`，改用 **PyGObject（Gst appsink）** 拉影格回傳 numpy，並包成與 cv2 相容的 `isOpened()/read()/release()` 介面，主迴圈不需改動。
- 容器只讀「已解碼的 BGR 影格」，完全**不需 Argus / EGL / nvidia 相機 lib**，徹底解耦。

### 4. 程式碼修改點

| 檔案 | 修改 |
|---|---|
| `src/pipeline/main.py` | 新增 `GstShmCapture`（PyGObject appsink）＋ `--source {csi,shm}` 切換（預設 `csi`，host 原生跑法不受影響） |
| `deploy/docker-compose.yml` | 加 `ipc: host`、掛 `/tmp/camshm` socket 目錄、`CAMERA_SOURCE=shm`、啟動指令 `--source shm`；移除無效的 `/dev/video0` |
| `deploy/start_camera_bridge.sh` | host 端 CSI → shmsink 擷取腳本（新檔，含清理與環境變數覆蓋） |

### 5. 複現步驟

```bash
# 1. host：啟動相機橋接（背景）
./deploy/start_camera_bridge.sh &
# 2. 容器：起完整 AI pipeline（mosquitto + access-control，自動 --source shm）
cd deploy && IMAGE_TAG=sha-d4bd675 docker compose up
```

解析度／socket 路徑由環境變數控制（`CAMERA_SHM_WIDTH` / `CAMERA_SHM_HEIGHT` / `CAMERA_SHM_FPS` / `CAMERA_SHM_SOCKET`），**host 與容器須設成一致**。

### 6. 記憶體開銷

shm 環形緩衝約 **20MB**（1280×720×3 × 8 幀），非 GB 等級。Jetson Orin Nano 8GB 共享記憶體的壓力來自 **AI 模型**（TensorRT engine + CUDA context），與相機方案無關——換相機方案不會改善模型佔用。

### 7. 方案比較

| 方案 | 容器內跑 AI | 教授複現難度 | 結論 |
|---|---|---|---|
| A 相機進容器 | — | EGLDisplay 硬牆、無底洞 | ❌ 不可行 |
| B pipeline 跑 host | ❌（跑 host） | 需 host 全套 PyTorch/TensorRT 依賴 | 可行但非容器化 |
| **C shm 橋接（採用）** | ✅ | `docker compose up` + 一行版本無關的 host 指令 | ✅ |

### 8. 已知限制

- host 仍需跑一個輕量擷取進程（`start_camera_bridge.sh`），並非「純容器」部署。
- 容器與 host 的影格解析度／socket 路徑須一致，否則 `shmsrc` 的 caps 對不上會讀不到影格。

---

## 測試情境設計與統計方法（Test Scenario Design）

> 本節重點不在「準確率數字」，而在**測試情境如何設計**：每個情境刻意只變動一個變因、隔離一個決策關卡，藉此驗證系統而非模型。

### 1. 設計原則

決策由四個關卡串成：**HC-SR04 距離 gate → 人臉偵測 → 身分比對（similarity≥0.5 且在 DB）→ 活體（連續幀）**。測試情境逐一隔離每個關卡，並加入「對照組」證明系統不會無差別判 SPOOF。

> 受測對象：**B（已註冊、可現場測）** 與 **C（臨時、未註冊）**。A 已在 DB 但本次無法現場 live 測——以下用 C（陌生人→UNKNOWN）與 S9（B 不被認成 A）即可驗證「DB 成員判斷」，A 無法 live 測列為限制。

### 2. 測試情境矩陣

| # | 情境 | 對象 | 輸入 | 預期決策 | 隔離的關卡 |
|---|---|---|---|---|---|
| S1 | 註冊者真人 | B | 真臉 @100cm | **GRANT** | happy path：比對+活體+連續4幀 |
| S2 | 註冊者照片 | B | 手機/列印 B 照片 | **SPOOF** | 活體擋平面照片 |
| S3 | 註冊者螢幕 | B | 筆電/手機螢幕播 B | **SPOOF** | 活體擋螢幕（反光/摩爾紋） |
| S4 | 陌生人真人 | C | 真臉 @100cm | **UNKNOWN** | DB 比對（不在庫） |
| S5 | 陌生人照片 | C | C 照片 | **SPOOF** | 優先序：活體 > unknown |
| S6 | 無人 | — | 空景/背景 | **IGNORE** | HC-SR04 距離 gate |
| S7 | 一閃而過 | B | 出現 <4 幀就離開 | **不開門** | 連續幀時間一致性 |
| S8 | 距離太遠 | B | 站 >100cm | **IGNORE** | 距離 gate 關閉 |
| S9 | 身分辨識正確性 | B | 真臉 | identity=**B**（非 A） | 兩個註冊者間的區分力 |
| S10 | 對照組：真人變裝 | B | 戴眼鏡/口罩/低光 | **非 SPOOF** | 證明系統不會無差別判假 |

### 3. 「什麼情況算 SPOOF」的判據

凡**2D 平面重現**皆應判 SPOOF：列印照片（霧面/亮面）、手機螢幕、筆電/平板螢幕、replay 影片——因缺乏 3D 深度／微動／背景一致性，活體分數偏低。反例（S10，real，**不該**判 SPOOF）：真人戴眼鏡/口罩/帽子/低光/側臉，應落在 GRANT/DENY/UNKNOWN。

### 4. 統計方式（雙層）

模型弱、逐幀很吵，故**同時報「逐幀分布」與「最終決策」**——系統可靠性來自時間聚合，非單幀準確率。

**第一層｜逐幀（原始模型能力）**：`similarity`、`live` 的 median / mean / std；逐幀過關率（`live≥0.3`、`similarity≥0.5` 的幀比例）。

**第二層｜決策（系統最終行為）**：GRANT/DENY/UNKNOWN/SPOOF/IGNORE 佔比；關鍵事件（是否開門、蜂鳴器次數）。

**彙總指標（混淆矩陣式）**

| 指標 | 定義 | 期望 |
|---|---|---|
| TAR | B 真人 → GRANT 比例 | 高 |
| **FAR（最關鍵）** | 照片/螢幕 → 誤開門比例 | **≈ 0** |
| Spoof recall | 照片/螢幕 → SPOOF 比例 | 高 |
| 誤報率 | B 真人 → 誤觸蜂鳴器 | ≈ 0 |
| UNKNOWN 正確率 | C 真人 → UNKNOWN 比例 | 高 |

### 5. 資料收集流程

用 `scripts/record_session.py` 包住主程式：它會在每個情境開始前**倒數緩衝**（避免站位/拿照片時的誤操作），並把「情境起訖」標記寫進同一份帶時間戳的 log，之後可依標記切出每段做統計。

```bash
# 包住主程式錄製（每段開始前有倒數緩衝）
python scripts/record_session.py -- pdm run python src/pipeline/main.py --no-display
#   依提示輸入情境編號(1=S1…) → 倒數 → 就定位 → 完成後按 Enter 結束該段
# 產出：logs/session_<時間>.log（決策+標記）、logs/session_<時間>.markers.jsonl（結構化起訖）
```

### 6. 結果表模板（填入實測）

| 情境 | 幀數 | sim median | live median | 決策分布 | 開門 | 蜂鳴器 | 判定 |
|---|---|---|---|---|---|---|---|
| S1 B真人 | | | | | | | |
| S2 B照片 | | | | | | | |
| S4 C真人 | | | | | | | |
| … | | | | | | | |

### 7. 已知限制

- 受測 live 對象僅 B + C，屬小樣本；本測試重在**情境設計的完整性**（涵蓋每個決策關卡與對照組），非統計顯著性。
- A 已在 DB 但無法現場 live 測，故「正確註冊者被拒/誤判」一類僅以 B 驗證。