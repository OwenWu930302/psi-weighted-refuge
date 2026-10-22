# 方法

## 1. 資料

| 項目 | 內容 |
|---|---|
| 資料集 | REFUGE（官方切分）：training 400、validation 400、testing 400 |
| 相機 | training：Zeiss Visucam 500（2124×2056）；validation / testing：Canon CR-2（1634×1634） |
| 模型輸入 | FunduSegmenter 提供的 800×800 視盤 ROI（`<split>/ROIs/images`），附 `centres.txt`（ROI 中心與原圖尺寸） |
| 標註 | `masks/*.bmp`：0 = 視杯、128 = 視盤環、255 = 背景 → 類別 2 / 1 / 0 |
| 結構 | OC = 類別 2；**OD = 類別 1 ∪ 2**（視盤環與視杯合併） |
| 病例分數 | (OD Dice + OC Dice) / 2，在原圖尺寸上計算 |

使用的 ROI 與 FunduSegmenter 官方 `Datasets.zip`（Hugging Face：JusticeZzy/FunduSegmenter）中 REFUGE 的 2,400 個檔案逐一比對完全相同（`diff -rq`）。

---

## 2. baseline 設定的反推與驗證

baseline 為先前以官方 FunduSegmenter 訓練的模型（`best_designed.pth`）。訓練指令沒有完整保存，以下每一項都由檔案反推並驗證。

| 項目 | 值 | 驗證方式 |
|---|---|---|
| 程式 | 官方 `train.py`，未修改 | `git diff -w`：只有檔案權限變更與刪除部分 baseline 檔案 |
| image_size | **256** | 以 epoch 197 權重照 `train.py` 的 validation 流程重算 mean Dice：256 為 0.9167485888，與 log 差 **0**；224 差 4.4e-04 |
| batch_size | **16** | 權重檔內 optimizer 的更新次數 4925 ÷ 最佳 epoch 197 = 25 = 400 ÷ 16 |
| epochs | 設定 10000（預設），**手動停於 1558** | log 進度條 `1558/10000`；log 無結束訊息 |
| 最佳 epoch | **197**（validation mean Dice 0.9167） | log 與權重檔的 `epoch` 欄位 |
| 驗證資料 | validation 400 張（`--separate_val 1`） | log：`Number of validation images: 400` |
| 資料增強 | `designed_transform` | log 檔名與原實驗紀錄 |
| 其他 | AdamW、lr 0.001、weight decay 0.001、seed 112316、Dice + CE | 官方 `train.py` 預設值 |
| 測試推論 | `test.py`，image_size 256 | 重算 testing 的 Dice、Ψ1、Ψ5 與原結果差 0；OD mean Dice 0.9394 與原 log 相同 |

與原論文（Zhao et al., arXiv:2508.11354）設定的差異：原論文 batch 32、最多 20,000 epoch、RTX 5090（32 GB）；本研究 batch 16（RTX 4080 16 GB 在 batch 32 時 GPU 記憶體不足）。其餘（256×256 輸入、AdamW lr 0.001 / weight decay 0.001、Dice + CE、seed 112316、以 validation 最高 DSC 選權重、800×800 ROI 預測後貼回原圖）相同。baseline 在 testing 的 OD / OC 為 93.94 / 85.61，原論文為 93.61 / 85.78（見 RESULTS 第 1 節）。

注意：原本計算 Ψ2 / Ψ8 的腳本（`run_psi2_psi8_refuge.py` v3）寫死 `IMAGE_SIZE = 224`，與模型的 256 不一致（見第 4.3 節）。

---

## 3. 專家模型與對照組

### 3.1 `train_psi.py` 的改動

`train_psi.py` 由官方 `train.py` 複製，所有改動以標記區分，完整差異見 `fundusegmenter_extension/train_psi.diff`（官方原行只有 4 行被替換）。

**[Ψ加權]**
- 新增 `--psi_cases`（清單 csv）與 `--psi_factor`（倍率，預設 3.0）。不給清單時所有權重為 1。
- 資料集外包一層 `PsiWeightedDataset`，不改變原資料與順序，只多回傳每張影像的權重。
- loss 改為逐張：`L = Σ wᵢ (Diceᵢ + CEᵢ) / Σ wᵢ`。Dice 與 CE 使用與原版相同的物件，只是 `reduction='none'`。
- **自我檢查**：第一個 batch 同時以原方式與「權重全為 1 的逐張方式」計算 loss，差異超過 1e-5 即停止訓練。實測差 0（1 epoch 測試）與 1.19e-07（正式訓練，GPU 浮點誤差）。
- 每個 epoch 記錄被加權的樣本數（batch 16 時每 epoch 恰為 120）。

**[記憶體]**
- `LightDatasetFromRoot`：mask 讀入時即取第 0 通道並以 uint8 保存。官方版以 800×800×3 int64 保存，800 張約 12 GB，在 15 GB 記憶體的機器上被 OOM killer 終止。
- 等價性：官方 `__getitem__` 最後只取 `mask[:,:,0]`；幾何轉換對 mask 為最近鄰、逐通道獨立；類別值 0/1/2 在 uint8 與 int64 下相同。`check_light_dataset.py` 以固定轉換與相同亂數種子逐張驗證送進模型的 (x, y) 完全相同。
- `--grad_ckpt`（預設 0，未使用）：encoder 逐 block gradient checkpointing，結果不變，供 GPU 記憶體不足時使用。

### 3.2 訓練設定

除以下三點外，與 baseline 完全相同（image_size 256、batch 16、AdamW、`designed_transform`、seed 112316、validation 400 張挑最佳 epoch）：

1. 清單病例 loss × 3
2. `--epochs 600`：baseline 的最佳 epoch 197 是前 1558 個 epoch 中的最大值，只看前 600 個時挑到的仍是 197；專家同樣在前 600 個中挑最佳，兩邊規則一致。三個專家的最佳 epoch（202、128、202）皆遠低於 600。
3. 輸出位置

### 3.3 對照組

以 `train_psi.py` 不給清單、其餘設定與專家完全相同，重新訓練一次（`run_control.sh`）。

FunduSegmenter 的 Dice loss 本來就逐張計算，因此不需要 psi-weighted-nnunet 的 1.0001 倍率。但對照組仍然必要，因為它提供「沒有加權、但重新訓練」的一次結果：

- 同設定重跑並不會得到相同的模型：同 seed 的第一個 batch loss 已不同（1.88713825 vs 1.89381289，GPU 運算不具確定性），且每張影像的 Dice 在不同次訓練之間浮動（第 7 節）。
- 加權清單由 baseline 的預測挑出，以 baseline 當「加權前」會混入回歸平均；專家與對照組都是新的一次訓練，回歸平均兩邊抵消。

---

## 4. Ψ 的計算

### 4.1 推論與還原（與 `test.py` 相同）

ROI → `validation_transform(256)` → 模型 → bicubic 插值到 800×800 → argmax → 依 `centres.txt` 貼回原圖尺寸 → 預測圖（255 背景、128 視盤環、0 視杯）。Dice 在原圖尺寸上計算。
`refuge_infer_psi.py` 逐張讀取影像，不預先載入整個資料集。

### 4.2 遮罩定義與病例層級

遮罩定義與 psi-weighted-nnunet `run_all_psi.load_refuge()` 相同：cup = 0、disc = 0 ∪ 128、rim = 128。

| Ψ | REFUGE 上的計算 |
|---|---|
| Ψ1 凸包比 | OC、OD 各自計算 |
| Ψ3 / Ψ9 包含 / 巢狀 | cup 落在 disc 外的比例（僅 OC） |
| Ψ4 鄰接 | cup 與 rim 的交集 / 聯集 |
| Ψ5 互斥 | OC：cup 與 rim 的重疊；OD：disc 與 cup 的重疊 |
| Ψ6 連通元件 | 各結構的元件數 − 1 |
| Ψ7 拓撲環 | rim 的內部洞數 − 1 |
| Ψ10 中心偏移 | 各結構質心與 disc 質心的距離 / 影像對角線 |
| Ψ2 / Ψ8 | 需重新推論（第 4.3 節），只比較視杯 |

病例層級：各結構取 nanmax（psi-weighted-nnunet `recompute_caselevel.to_case`）。
`refuge_select_proxy.py` 直接 import 上述倉庫的 Ψ 函式、`to_case` 與 `report`，不另行實作。

**Ψ5 在 REFUGE 上的意義**：cup 與 rim 互斥（交集為 0），而 disc 包含 cup，因此 OD 的 Ψ5 = |cup| / |disc|，取 nanmax 後 Ψ5 ≈ **預測的視杯面積 ÷ 視盤面積**。Ψ5 取最低 30% 即挑出預測視杯最小的影像。

### 4.3 Ψ2 / Ψ8：擾動施加於 ROI 輸入

REFUGE 的推論依 `centres.txt` 的絕對座標裁出 ROI，旋轉或遮擋原圖會使座標失效。因此擾動施加於模型輸入（已裁切、縮放、正規化的 ROI），與原 v3 腳本相同：

- Ψ8：輸入旋轉 15°（雙線性）→ 推論 → 預測轉回（最近鄰）→ 1 − Dice（視杯）
- Ψ2：於預測視杯質心處施加垂直亮度衰減帶（參數與 CAMUS 相同：A = 3.0、帶寬 25%、邊緣 15%）→ 推論 → 1 − Dice（視杯）
- 推論使用 AMP、插值到 800×800（v3 檔頭說明：移除插值會使視杯消失）

**輸入尺寸由 224 改為 256**：原 v3 以 224 計算，與模型訓練及 Dice、Ψ1、Ψ5 使用的 256 不一致。本研究統一改用 256。
在 testing 上，Ψ8 改為 256 後仍顯著且更強（AUC 0.575 → 0.591，p 0.017 → 0.0038）；兩種尺寸的排名相關 ρ = 0.90。

### 4.4 重現驗證

| 項目 | 驗證 | 結果 |
|---|---|---|
| Dice（OD、OC、病例） | testing 400 張 vs 原 `per_image_dice_designed.csv` | 最大差 < 5e-07 |
| Ψ1、Ψ5 | testing vs 原 `caselevel_refuge.csv` | 差 0，ρ = 1.000 |
| Ψ2、Ψ8（以 224 計算） | testing 20 張 vs 原 v3 輸出 | 差 0，ρ = 1.000 |
| 整套 proxy 檢定 | 以本倉庫程式重算 testing 表 | 與原表逐列相同 |

---

## 5. proxy 篩選

規則與 psi-weighted-nnunet 相同：病例層級，低分組 = Dice 最低 30%（含並列），Mann-Whitney 檢定，p < 0.05 為顯著。

**篩選資料為 validation 集**（baseline 推論，400 張）。validation 未用於訓練，只用於挑選最佳 epoch；測試集在整個篩選與訓練過程中都未使用。

| proxy | validation AUC | p | testing AUC（參考） | p |
|---|---|---|---|---|
| Ψ1 凸包比 | **0.689** | 1.9e-09 | 0.689 | 1.9e-09 |
| Ψ2 亮度衰減 | 0.537 | 0.24 | 0.479 † | 0.52 |
| Ψ5 互斥（反向） | **0.319** | 9.0e-09 | 0.301 | 3.0e-10 |
| Ψ6 連通元件 | 恆定 | — | 0.496 | 0.36 |
| Ψ8 旋轉一致性 | **0.631** | 3.1e-05 | 0.591 | 3.8e-03 |
| Ψ10 中心偏移 | 0.562 | 0.0498 | 0.488 | 0.70 |
| Ψ3 / Ψ4 / Ψ7 / Ψ9 | 恆定 | — | 恆定 | — |

† testing 的 Ψ2 為原 v3 以 224 計算；其餘 testing 數值為 256。

選用 **Ψ1、Ψ5、Ψ8**。Ψ10 在 validation 上僅勉強達 0.05 且在 testing 上不顯著，未訓練專家。

**時序說明**：加權清單最初依 testing 的篩選結果（Ψ1、Ψ5、Ψ8）產生；之後改以 validation 重新篩選，選出的 proxy 相同，因此清單與專家不需更動，但測試集未參與 proxy 選擇。

---

## 6. 加權清單

由 baseline 對 training 400 張的預測計算 Ψ（模型訓練時看過這些影像），規則與 psi-weighted-nnunet `make_weight_lists.select()` 相同：

| 專家 | 規則 | 門檻 | 例數 | 清單內 Dice | 清單外 Dice |
|---|---|---|---|---|---|
| Ψ1 | 最高 30% | ≥ 1.00818 | 120 | 0.9722 | 0.9766 |
| Ψ5 | 最低 30%（反向） | ≤ 0.1968 | 120 | 0.9719 | 0.9768 |
| Ψ8 | 最高 30% | ≥ 0.05136 | 120 | 0.9731 | 0.9763 |

清單重疊：Ψ1 ∩ Ψ5 = 77、Ψ1 ∩ Ψ8 = 49、Ψ5 ∩ Ψ8 = 50。

---

## 7. 評估

**訓練集（主要分析）**：與 psi-weighted-nnunet `weighted_cases_same_loss.py` 相同的格式。
- 被加權病例：加權前 / 後 Dice、Δ、升 / 降例數、Wilcoxon
- 加權組 Δ − 未加權組 Δ（以未加權組為參考線，扣掉兩模型的整體偏移）、Mann-Whitney
- 加權前 = 對照組（主要）；對 baseline 的結果作為補充
- 另以「同一清單在其他模型中未被加權時的差距」作為該清單病例本身的傾向，對角線減去它估計加權本身的效果

**安慰劑檢查**（`refuge_placebo.py`）：
1. 交叉清單：在模型 M 中，以別的清單中「在 M 裡沒被加權」的病例計算差距；若與真正被加權的差距相近，效果不是加權造成的
2. 控制 baseline Dice：依 baseline Dice 分五組，只在組內比較加權與未加權，分層置換檢定 10,000 次；另以線性迴歸 Δ ~ 加權 + baseline Dice 確認
3. 回歸平均：未加權病例中 Δ 對 baseline Dice 的斜率

**測試集**：所有實驗完成後評估一次（與 BTCV、ACDC 相同的時點）。

---

## 8. 已知限制

1. **選擇模型的方式**：FunduSegmenter 只保存 validation 最佳的 epoch，各模型停在不同 epoch（baseline 197、專家 202 / 128 / 202），對訓練集 Dice 的影響可能大於加權本身。nnU-Net 則使用固定的最後一個 epoch。
2. **訓練集接近飽和**：baseline 在訓練集的病例 Dice 為 0.9753，被加權病例 0.972，改善空間小。
3. **凍結的 encoder**：可學習參數僅 36M（共 340M），針對個別病例特化的能力有限。
4. **清單重疊**：Ψ1 與 Ψ5 的清單重疊 64%，兩個專家不完全獨立。
5. **相機差異**：training 為 Zeiss，validation / testing 為 Canon。
6. **Ψ 的意義**：REFUGE 上 Ψ5 反映視杯相對大小（解剖類型），不一定代表模型錯誤；Ψ1 的數值範圍極窄（約 1.008–1.010）。
7. **重跑變異**：對照組與 baseline 設定相同，最佳 epoch 為 347 與 197，訓練集 Dice 0.9777 與 0.9753，測試集 0.8972 與 0.8978。每個模型只訓練一次；Wilcoxon 以影像為單位，不包含不同次訓練之間的變異。
8. **未檢驗的對照**：沒有以隨機清單（任意 120 例 ×3）訓練，無法區分測試集的提升來自「Ψ 選到的病例」還是任何形式的加權。
