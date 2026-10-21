# 完整重現步驟

每一步都附檢查指令。兩個 conda 環境：

- `fundusegmenter`：推論與訓練（FunduSegmenter 官方環境）
- `expertree`：只讀 CSV 的分析（psi-weighted-nnunet 的環境，見 `requirements_analysis.txt`）

---

## 0. 硬體與時間

| 項目 | 本研究 |
|---|---|
| GPU | RTX 4080 16 GB（batch 32 會 GPU OOM，baseline 與專家皆用 batch 16） |
| 記憶體 | 15 GB（官方資料載入方式會 OOM，見第 4 步） |
| 每個模型 | 600 epoch 約 5.6 小時（每 epoch 約 34 秒） |
| 推論 | 每張約 0.3 秒（含 Ψ2 / Ψ8 約 0.6 秒） |

---

## 1. 環境與程式

```bash
# 1-1 FunduSegmenter（官方）
git clone https://github.com/JusticeZzy/FunduSegmenter
cd FunduSegmenter && git checkout db74034
conda create -n fundusegmenter python=3.11.0 -y
conda activate fundusegmenter
pip install -r requirements.txt
# 下載 RETFound_mae_natureCFP.pth（見 FunduSegmenter README），放在 FunduSegmenter/

# 1-2 本倉庫與 psi-weighted-nnunet（分析程式會 import 後者的 Ψ 函式）
git clone https://github.com/OwenWu930302/psi-weighted-refuge.git
git clone https://github.com/OwenWu930302/psi-weighted-nnunet.git

# 1-3 把本倉庫的程式放進 FunduSegmenter 資料夾
cp psi-weighted-refuge/fundusegmenter_extension/* psi-weighted-refuge/analysis/* FunduSegmenter/
```

檢查：

```bash
cd FunduSegmenter
diff train.py train_psi.py | diff - train_psi.diff && echo "train.py 為官方版本"
python -c "import torch, monai, albumentations; print(torch.__version__, monai.__version__, albumentations.__version__)"
```

---

## 2. 資料

REFUGE 原始資料（training / validation / testing 各含 `images/`、`masks/`）與 FunduSegmenter 處理好的 ROI（`ROIs/images`、`ROIs/masks`、`ROIs/centres.txt`），可由 FunduSegmenter 的 `Datasets.zip` 取得（Hugging Face：JusticeZzy/FunduSegmenter）。

```bash
export REFUGE_DATA=<路徑>/REFUGE          # 底下為 training/ validation/ testing/
mkdir -p ../psi_expert                     # 所有輸出（位於 FunduSegmenter 的上一層）
```

檢查：

```bash
for s in training validation testing; do
  echo "$s: images $(ls $REFUGE_DATA/$s/images | wc -l)  masks $(ls $REFUGE_DATA/$s/masks | wc -l)  ROI $(ls $REFUGE_DATA/$s/ROIs/images | wc -l)"
done   # 皆應為 400
```

以下指令中的 `--data_root` 預設為 `/mnt/data2t/datasets/REFUGE/REFUGE`，資料在別處時加上 `--data_root $REFUGE_DATA`。

---

## 3. baseline

### 3-1 訓練（官方 `train.py`）

```bash
MPLBACKEND=Agg python train.py --epochs 10000 --batch_size 16 \
  --model_selection FunduSegmenter --output_channel 3 --label_n_cls 3 --seed 112316 \
  --train_image_path $REFUGE_DATA/training/ROIs/images --train_mask_path $REFUGE_DATA/training/ROIs/masks \
  --separate_val 1 --val_image_path $REFUGE_DATA/validation/ROIs/images --val_mask_path $REFUGE_DATA/validation/ROIs/masks \
  --num_workers 1 --image_size 256 --transform_mode designed_transform
```

原實驗於 epoch 1558 手動停止，最佳 epoch 197。權重存於 `results/saved_weights/best_weights.pth`，本研究另存為 `repro_notes/best_designed.pth`。
（官方資料載入會使用約 14 GB 記憶體；記憶體不足時以 `train_psi.py` 不給清單訓練，結果等價，見第 4 步。）

### 3-2 確認設定（以既有權重驗證）

```bash
python3 check_train_size.py        # image_size 256 應標示「← 吻合」（與 log 差 0）
python3 -c "
import torch; ck = torch.load('repro_notes/best_designed.pth', map_location='cpu', weights_only=False)
s = ck['optimizer']['state']; step = int(next(iter(s.values()))['step'])
print('epoch', ck['epoch'], ' 每個 epoch 更新', step / ck['epoch'], '次（25 = batch 16）')"
```

### 3-3 推論：testing / validation / training

```bash
python3 refuge_infer_psi.py --split testing    --image_size 256 --out ../psi_expert/testing256
python3 refuge_infer_psi.py --split validation --image_size 256 --out ../psi_expert/validation256
python3 refuge_infer_psi.py --split training   --image_size 256 --out ../psi_expert/baseline_training
```

每個輸出資料夾含 `per_image.csv`（Dice、Ψ1、Ψ5、Ψ2、Ψ8）與 `map/`（預測圖）。

### 3-4 檢查：與原結果一致

```bash
# Dice / Ψ1 / Ψ5 用 256、Ψ2 / Ψ8 用原 v3 的 224 → 應與原 caselevel_refuge.csv 全部一致
python3 refuge_infer_psi.py --split testing --image_size 256 --psi8_size 224 --limit 20 --out ../psi_expert/check_psi2
conda run -n expertree python3 refuge_lists.py check --new ../psi_expert/check_psi2/per_image.csv \
  --old_case <原 caselevel_refuge.csv> --old_dice repro_notes/per_image_dice_designed.csv
```

預期：全部 ✓。

---

## 4. 驗證 `train_psi.py`

```bash
python3 check_light_dataset.py      # 省記憶體載入：送進模型的 (x, y) 與官方相同，全部 ✓
```

1 epoch 測試（約 3 分鐘）：

```bash
MPLBACKEND=Agg python3 train_psi.py --epochs 1 --batch_size 16 \
  --model_selection FunduSegmenter --output_channel 3 --label_n_cls 3 --seed 112316 \
  --train_image_path $REFUGE_DATA/training/ROIs/images --train_mask_path $REFUGE_DATA/training/ROIs/masks \
  --separate_val 1 --val_image_path $REFUGE_DATA/validation/ROIs/images --val_mask_path $REFUGE_DATA/validation/ROIs/masks \
  --num_workers 1 --image_size 256 --transform_mode designed_transform \
  --results_base_dir ../psi_expert/smoke \
  --psi_cases ../psi_expert/lists/wcases_refuge_psi1_convex.csv --psi_factor 3.0 \
  > ../psi_expert/smoke/train.log 2>&1
grep -E "Ψ加權|Epoch_1 weighted|best weights" ../psi_expert/smoke/train.log
```

預期：`被加權 120 / 400 張`、自我檢查差 ≤ 1e-6、`Epoch_1 weighted samples: 120`。（需先完成第 6 步產生清單。）

---

## 5. proxy 篩選（validation）

```bash
REPO=<psi-weighted-nnunet 路徑>
conda run -n expertree python3 refuge_select_proxy.py --pred ../psi_expert/validation256 --repo $REPO
```

預期（最低 30%）：Ψ1、Ψ5（反向）、Ψ8 顯著，Ψ10 邊緣。數值見 [RESULTS.md](RESULTS.md) 第 2 節。

---

## 6. 加權清單

```bash
conda run -n expertree python3 refuge_lists.py make \
  --table ../psi_expert/baseline_training/per_image.csv --out ../psi_expert/lists
```

預期：三份清單各 120 例；門檻 Ψ1 ≥ 1.00818、Ψ5 ≤ 0.1968、Ψ8 ≥ 0.05136（與 `weight_lists/` 相同）。

---

## 7. 專家模型與對照組

```bash
nohup systemd-inhibit --why="REFUGE experts" bash run_refuge_experts.sh > ../psi_expert/run_experts.out 2>&1 &
# 完成後：
nohup systemd-inhibit --why="REFUGE control" bash run_control.sh > ../psi_expert/run_control.out 2>&1 &
```

兩者不可同時執行（GPU 記憶體）。檢查進度：

```bash
grep "=====" ../psi_expert/run_experts.out
grep -o "[0-9]*/600" ../psi_expert/expert_psi1_convex/train.log | tail -1
grep -E "best weights|Training time" ../psi_expert/*/train.log
```

---

## 8. 訓練集分析

```bash
# 8-1 專家與對照組對 training 400 張推論（不需 Ψ2 / Ψ8）
for M in expert_psi1_convex expert_psi5_exclusion expert_psi8_rotation control; do
  python3 refuge_infer_psi.py --split training --image_size 256 --no_psi8 \
    --ckpt ../psi_expert/$M/saved_weights/best_weights.pth --out ../psi_expert/${M}_training
done

# 8-2 加權前後比較（主要：對對照組；補充：對 baseline）
conda run -n expertree python3 refuge_compare.py --root ../psi_expert --before control
conda run -n expertree python3 refuge_compare.py --root ../psi_expert

# 8-3 安慰劑檢查
conda run -n expertree python3 refuge_placebo.py --root ../psi_expert --before control
conda run -n expertree python3 refuge_placebo.py --root ../psi_expert
```

---

## 9. 測試集（所有實驗完成後）

```bash
for M in expert_psi1_convex expert_psi5_exclusion expert_psi8_rotation control; do
  python3 refuge_infer_psi.py --split testing --image_size 256 --no_psi8 \
    --ckpt ../psi_expert/$M/saved_weights/best_weights.pth --out ../psi_expert/${M}_testing
done
conda run -n expertree python3 refuge_compare.py --root ../psi_expert --split testing --before control
conda run -n expertree python3 refuge_compare.py --root ../psi_expert --split testing
```

測試集沒有被加權的病例，請看第二張表的「整體Δ」與「整體p」。

---

## 10. 收集結果到倉庫

```bash
cd psi-weighted-refuge
PSI_EXPERT=<psi_expert 路徑> FS=<FunduSegmenter 路徑> bash tools/collect_results.sh
git add -A && git status
```
