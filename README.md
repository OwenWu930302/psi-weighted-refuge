# Ψ-Weighted Expert Models on REFUGE（FunduSegmenter）

**把 [psi-weighted-nnunet](https://github.com/OwenWu930302/psi-weighted-nnunet) 的 Ψ 加權專家流程，套用到 REFUGE 視盤／視杯分割與 FunduSegmenter（RETFound）模型。**

Ψ 的定義、proxy 篩選、加權清單規則、加權 loss 公式，全部沿用 psi-weighted-nnunet；
差別在分割模型（FunduSegmenter，而非 nnU-Net）與資料集（REFUGE 眼底彩照）。
BTCV 與 ACDC 的結果請見原倉庫。

---

## 流程

```
① baseline（官方 FunduSegmenter，原始訓練設定）
        │
        ▼
② baseline 推論 validation 400 張 → 計算 Ψ1–Ψ10 → proxy 篩選（不使用測試集）
        │   選出 Ψ1 凸包比、Ψ5 互斥、Ψ8 旋轉一致性
        ▼
③ baseline 推論 training 400 張 → 依 proxy 取 30% 病例 → 加權清單（各 120 例）
        │
        ▼
④ 專家模型：清單病例 loss × 3，其餘不變             ⑤ 對照組：同設定、不加權，重新訓練一次
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
⑥ 訓練集：被加權病例在「專家 vs 對照組」下的 Dice 變化
⑦ 測試集：所有實驗完成後才評估一次
```

$$L = \frac{\sum_i w_i \cdot L_{\text{Dice+CE}}(x_i)}{\sum_i w_i}, \qquad w_i \in \{1,\ 3\}$$

---

## 與 psi-weighted-nnunet 的差異

| | psi-weighted-nnunet（BTCV / ACDC） | 本倉庫（REFUGE） |
|---|---|---|
| 分割模型 | nnU-Net v2.8.1（整個網路訓練） | FunduSegmenter：RETFound ViT-L 凍結（311M），只訓練 adapter / decoder（36M） |
| 輸入 | 2D 切片 | 以視盤為中心的 800×800 ROI，縮放至 256×256，預測後依 `centres.txt` 貼回原圖 |
| 選用的權重 | 最後一個 epoch | validation mean Dice 最佳的 epoch（FunduSegmenter 預設、唯一的存檔方式） |
| loss | CE + Dice；官方整批計算 Dice | Dice + CE；monai DiceLoss 本來就逐張計算 |
| 對照組 | 倍率 1.0001（讓 loss 也逐張計算） | 不給清單（權重全為 1）；loss 已逐張，不需 1.0001 |
| proxy 篩選 | 5 折 OOF（BTCV、ACDC Ψ10）／測試集（ACDC Ψ1/6/7 †） | **validation 集**（官方切分，未用於訓練） |

---

## 進度

| 步驟 | 狀態 |
|---|---|
| baseline 設定反推與驗證 | 完成 |
| Ψ1–Ψ10 計算（重現原結果，差異 0） | 完成 |
| proxy 篩選（validation） | 完成：Ψ1、Ψ5、Ψ8 |
| 加權清單 | 完成 |
| 三個專家模型 | 完成 |
| 訓練集分析（對 baseline）＋ 安慰劑檢查 | 完成（初步） |
| 對照組 | **訓練中** |
| 訓練集分析（對對照組） | 待對照組完成 |
| 測試集評估 | 所有實驗完成後 |

---

## 主要設定

| | baseline | 專家 / 對照組 |
|---|---|---|
| 程式 | 官方 `train.py`（未修改） | `train_psi.py`（官方 `train.py` 加上加權；見 `train_psi.diff`） |
| image_size | 256 | 256 |
| batch_size | 16 | 16 |
| epochs | 設定 10000，手動停於 1558 | 600 |
| 最佳 epoch | 197（val mean Dice 0.9167） | Ψ1 202、Ψ5 128、Ψ8 202 |
| optimizer | AdamW，lr 0.001，weight decay 0.001 | 相同 |
| 資料增強 | `designed_transform` | 相同 |
| seed | 112316 | 112316 |
| 加權 | — | 清單病例 ×3（對照組：無） |

baseline 的每一項設定都由檔案反推並驗證，見 [docs/METHODS.md](docs/METHODS.md) 第 2 節。

---

## 目前結果（訓練集，初步）

**未控制回歸平均時**，被加權病例比未加權病例「掉得少」，Ψ5 顯著（+0.0049，p = 4.5e-05）。
**以 baseline Dice 分層控制回歸平均後**，三個專家都沒有顯著的加權效果（Ψ1 −0.0015、Ψ5 +0.0023、Ψ8 −0.0016）。

原因是「加權前」用的是產生清單的同一個 baseline：清單病例的 baseline Dice 本來就偏低，換一個模型重新預測就會回升（未加權病例中 Δ 對 baseline Dice 的斜率 −0.66 至 −0.89）。
對照組（訓練中）完成後，以「專家 − 對照組」重新比較，回歸平均會兩邊抵消。

完整數據見 [docs/RESULTS.md](docs/RESULTS.md)。

---

## 倉庫結構

```
fundusegmenter_extension/   需放進 FunduSegmenter 資料夾執行的程式（會 import 其 model/、utils/）
  train_psi.py              加權訓練（官方 train.py ＋ [Ψ加權] ＋ [記憶體]）
  train_psi.diff            train_psi.py 相對官方 train.py 的全部改動
  refuge_infer_psi.py       推論 → 每張 Dice、Ψ1、Ψ5、Ψ2、Ψ8、預測圖
  run_refuge_experts.sh     依序訓練三個專家
  run_control.sh            訓練對照組
  check_*.py                三項驗證（載入方式、訓練尺寸、gradient checkpointing）
analysis/                   只讀 CSV 的分析（不需 GPU）
  refuge_select_proxy.py    proxy 篩選（直接 import psi-weighted-nnunet 的 Ψ 函式與檢定）
  refuge_lists.py           與原結果比對（check / proxy）、產生加權清單（make）
  refuge_compare.py         被加權病例的加權前後比較
  refuge_placebo.py         安慰劑檢查（交叉清單、控制 baseline Dice、回歸平均）
tools/collect_results.sh    把本機結果複製進 results/、weight_lists/
weight_lists/               三份加權清單
results/                    每張影像的結果、proxy 表、比較、訓練曲線（不含影像與權重）
docs/                       方法、重現步驟、程式說明、結果
```

| 文件 | 內容 |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | 資料、baseline 設定的反推、專家設定、Ψ 計算、proxy 篩選、加權清單、已知限制 |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 從零開始的完整步驟與每一步的檢查 |
| [docs/CODE.md](docs/CODE.md) | 每支程式的用途、輸入輸出、與原程式的對應 |
| [docs/RESULTS.md](docs/RESULTS.md) | 所有數據與解讀 |

---

## 環境

FunduSegmenter（[JusticeZzy/FunduSegmenter](https://github.com/JusticeZzy/FunduSegmenter)，commit `db74034`）、PyTorch 2.4.1（CUDA 12.1）、monai 1.5.1、albumentations 2.0.8、單張 RTX 4080（16 GB）、記憶體 15 GB。
分析程式使用 psi-weighted-nnunet 的 `expertree` 環境（`requirements_analysis.txt`）。

---

## 致謝與授權

FunduSegmenter was created by Zhenyi Zhao of the University of Dundee, UK.

- Zhao, Z., Mookiah, M. R. K., & Trucco, E. (2026). FunduSegmenter: Leveraging the RETFound Foundation Model for Joint Optic Disc and Optic Cup Segmentation in Retinal Fundus Images. *Translational Vision Science & Technology*, 15(5), 14.
- Zhou, Y. et al. A foundation model for generalizable disease detection from retinal images (RETFound). *Nature* 622, 156–163 (2023).
- Orlando, J. I. et al. REFUGE Challenge. *Medical Image Analysis* 59, 101570 (2020).

本倉庫自撰的程式為 Apache-2.0（見 [LICENSE](LICENSE)）。
`train_psi.py` 衍生自 FunduSegmenter 的 `train.py`，保留其原始版權聲明；FunduSegmenter 與 RETFound 的權重為 CC BY-NC 4.0，僅限非商業用途，本倉庫不包含任何權重與影像。詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
