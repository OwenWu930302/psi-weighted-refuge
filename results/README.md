# results/

由 `tools/collect_results.sh` 從本機的 `psi_expert/` 複製而來（只含 CSV 與文字摘要）；`weighted_cases/` 由 `analysis/weighted_cases_same_loss.py` 產生。

| 資料夾 | 內容 | |
|---|---|---|
| `weighted_cases/` | **加權病例分析**：被加權病例的加權前（對照組）/ 後（專家），格式同 psi-weighted-nnunet | 主要 |
| `per_image/` | 每個模型 × split 的每張影像 Dice（OD、OC、病例）與 Ψ | 原始資料 |
| `proxy/` | 病例層級 Ψ 表（validation 用於篩選；testing 供參考） | |
| `curves/`、`logs/` | 訓練曲線與 log 摘要 | |
| `compare_testing_vs_control/`、`compare_testing/` | 測試集：各模型對對照組 / baseline | |
| `compare_training_vs_control/`、`compare_training/` | 訓練集補充分析（含未加權病例、安慰劑矩陣） | 補充 |

數據解讀見 [docs/RESULTS.md](../docs/RESULTS.md)。
