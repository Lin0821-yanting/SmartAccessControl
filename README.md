
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