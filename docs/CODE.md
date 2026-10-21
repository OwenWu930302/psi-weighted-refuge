# 程式說明

所有程式的檔頭都有用法與說明；本頁列出用途、輸入輸出，以及與既有程式的對應關係。

## fundusegmenter_extension/（放進 FunduSegmenter 資料夾執行）

| 程式 | 用途 | 輸入 → 輸出 | 對應的既有程式 |
|---|---|---|---|
| `train_psi.py` | 加權訓練（專家、對照組） | ROI 影像、加權清單 → `<results_base_dir>/saved_weights/best_weights.pth`、log | FunduSegmenter `train.py`（改動見 `train_psi.diff`） |
| `refuge_infer_psi.py` | 推論、Dice、Ψ1 / Ψ5 / Ψ2 / Ψ8、預測圖 | 權重、ROI、`centres.txt`、原圖標註 → `per_image.csv`、`map/*.png` | 推論與還原 = `test.py`；Ψ1 / Ψ5 = psi-weighted-nnunet `run_all_psi.py`；Ψ2 / Ψ8 = `run_psi2_psi8_refuge.py` v3 |
| `run_refuge_experts.sh` | 依序訓練 Ψ1、Ψ5、Ψ8 專家 | 清單 → `psi_expert/expert_<proxy>/` | |
| `run_control.sh` | 訓練對照組（不給清單） | → `psi_expert/control/` | psi-weighted-nnunet 的 ps1ctrl / allctrl |
| `check_light_dataset.py` | 驗證省記憶體載入與官方輸出相同 | 前 40 張 → ✓ / ✗ | `utils/datasets.py` `DatasetFromRoot` |
| `check_train_size.py` | 由 epoch 197 的 validation Dice 反推訓練時的 image_size | 權重、log → 各尺寸與 log 的差 | `train.py` 的 validation 迴圈 |
| `check_grad_ckpt.py` | 驗證 `--grad_ckpt` 不改變 loss 與梯度（本研究未使用） | 前 8 張 → ✓ / ✗、GPU 記憶體估計 | |

### `train_psi.py` 的改動（標記 `[Ψ加權]`、`[記憶體]`）

| 位置 | 改動 |
|---|---|
| 參數 | `--psi_cases`、`--psi_factor`、`--grad_ckpt` |
| `LightDatasetFromRoot` | mask 以單通道 uint8 保存（輸出不變） |
| `PsiWeightedDataset` | 包裝原資料集，多回傳每張權重；檢查清單病例都在訓練集 |
| loss | `reduction='none'` 的 Dice 與 CE，`Σ w L / Σ w`；第一個 batch 自我檢查 |
| log | `[Ψ加權] 清單 …`、`[Ψ加權] 自我檢查 …`、`Epoch_N weighted samples: …` |

官方原行只有 4 行被替換（兩個 Dataset 的建立、訓練迴圈的解包、loss 的計算），其餘皆為新增。

### `refuge_infer_psi.py` 的參數

| 參數 | 說明 |
|---|---|
| `--split` | training / validation / testing |
| `--ckpt` | 權重（預設 `repro_notes/best_designed.pth`） |
| `--image_size` | 模型輸入尺寸（本研究 256） |
| `--psi8_size` | Ψ2 / Ψ8 的輸入尺寸；0 = 同 `--image_size`（原 v3 為 224，僅供重現驗證） |
| `--no_psi8` | 不算 Ψ2 / Ψ8（專家評估用） |
| `--no_maps` | 不存預測圖（`refuge_select_proxy.py` 需要預測圖） |
| `--data_root` | REFUGE 資料根目錄 |

## analysis/（只讀 CSV，不需 GPU）

| 程式 | 用途 | 與既有程式的關係 |
|---|---|---|
| `refuge_select_proxy.py` | proxy 篩選：10 個 Ψ → 病例層級 → 檢定 | 直接 import psi-weighted-nnunet 的 `run_all_psi`（Ψ 函式）、`recompute_caselevel.to_case`、`report`；遮罩迴圈與 `load_refuge()` 相同 |
| `refuge_lists.py check` | 與原 `caselevel_refuge.csv`、`per_image_dice_designed.csv` 比對 | |
| `refuge_lists.py proxy` | 以原表為底、只替換 Ψ8 後重新檢定 | 呼叫 `recompute_caselevel.report` |
| `refuge_lists.py make` | 產生加權清單 | `select()` 逐字複製自 `make_weight_lists.py` |
| `refuge_compare.py` | 被加權病例的加權前後、加權組 vs 未加權組 | 輸出格式同 `weighted_cases_same_loss.py`；`--before baseline / control` |
| `refuge_placebo.py` | 交叉清單安慰劑、控制 baseline Dice（分層置換）、回歸平均斜率 | |

## tools/

| 程式 | 用途 |
|---|---|
| `collect_results.sh` | 把 `psi_expert/` 的 CSV、清單、訓練曲線、log 摘要複製進 `results/` 與 `weight_lists/`（不含影像、預測圖、權重） |

## results/ 的內容

| 路徑 | 內容 |
|---|---|
| `per_image/` | 每個模型 × split 的每張影像 Dice（OD、OC、病例）與 Ψ |
| `proxy/caselevel_refuge_validation.csv` | validation 病例層級 10 個 Ψ（用於篩選） |
| `proxy/caselevel_refuge_testing_original.csv` | 原 testing 表（Ψ2 / Ψ8 為 224） |
| `proxy/caselevel_refuge_testing_psi8at256.csv` | 同上，Ψ8 換成 256 |
| `compare_training*/` | 加權前後比較、安慰劑矩陣（`_vs_control` 為對對照組） |
| `curves/` | 每個 epoch 的 training loss、validation OD / OC / mean Dice、加權樣本數 |
| `logs/` | 各訓練 log 的關鍵行（張數、清單、自我檢查、最佳 epoch、訓練時間） |
