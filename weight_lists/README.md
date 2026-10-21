# weight_lists/

三個專家實際使用的加權清單（由 `tools/collect_results.sh` 複製）。
欄位：`case`（training 影像編號）、proxy 值、baseline 病例 Dice。產生方式見 [docs/METHODS.md](../docs/METHODS.md) 第 6 節。

| 檔案 | 規則 | 門檻 | 例數 |
|---|---|---|---|
| `wcases_refuge_psi1_convex.csv` | Ψ1 最高 30% | ≥ 1.00818 | 120 |
| `wcases_refuge_psi5_exclusion.csv` | Ψ5 最低 30% | ≤ 0.1968 | 120 |
| `wcases_refuge_psi8_rotation.csv` | Ψ8 最高 30% | ≥ 0.05136 | 120 |
