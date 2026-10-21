#!/bin/bash
# ============================================================
# REFUGE 對照組：與專家完全相同的設定，唯一差別是不給加權清單（所有權重 = 1）
#
# 作用：一次「沒有加權、但重新訓練」的模型。
#   同設定重跑本身就會讓每張影像的 Dice 浮動（GPU 運算不具確定性）；
#   專家 − 對照組 才能扣掉這個浮動與回歸平均，只剩加權本身的效果
#   （對應 psi-weighted-nnunet 的 ps1ctrl / allctrl）。
#   FunduSegmenter 的 Dice loss 本來就逐張計算，因此不需要 1.0001 的倍率。
#
# 用法（在 FunduSegmenter 資料夾內）：
#   conda activate fundusegmenter
#   nohup systemd-inhibit --why="REFUGE control" bash run_control.sh > ../psi_expert/run_control.out 2>&1 &
# ============================================================
set -euo pipefail
D=${REFUGE_DATA:-/mnt/data2t/datasets/REFUGE/REFUGE}
OUT=${PSI_EXPERT:-$(cd .. && pwd)/psi_expert}
export MPLBACKEND=Agg
R=$OUT/control
mkdir -p "$R"
echo "===== $(date '+%F %T') 開始 control ====="
python3 train_psi.py \
  --epochs 600 --batch_size 16 \
  --model_selection FunduSegmenter --output_channel 3 --label_n_cls 3 --seed 112316 \
  --train_image_path $D/training/ROIs/images   --train_mask_path $D/training/ROIs/masks \
  --separate_val 1 \
  --val_image_path   $D/validation/ROIs/images --val_mask_path   $D/validation/ROIs/masks \
  --num_workers 1 --image_size 256 --transform_mode designed_transform \
  --results_base_dir "$R" --grad_ckpt 0 \
  2>&1 | tee "$R/train.log"
echo "===== $(date '+%F %T') 完成 control ====="
