#!/bin/bash
# ============================================================
# REFUGE 專家模型：Ψ1 / Ψ5 / Ψ8 依序訓練（RTX 4080 每個約 5.6 小時）
#
# 用法（在 FunduSegmenter 資料夾內）：
#   conda activate fundusegmenter
#   nohup bash run_refuge_experts.sh > ../psi_expert/run_experts.out 2>&1 &
#   tail -f ../psi_expert/run_experts.out
#
# 除了 --psi_cases / --psi_factor / --epochs / 輸出位置之外，
# 其餘參數必須與 baseline 完全相同（輸入尺寸 256 已由 check 步驟確認）。
# ============================================================
set -euo pipefail
D=${REFUGE_DATA:-/mnt/data2t/datasets/REFUGE/REFUGE}   # REFUGE 資料根目錄（含 training/validation/testing）
OUT=${PSI_EXPERT:-$(cd .. && pwd)/psi_expert}         # 所有輸出的根目錄
LISTS=$OUT/lists
# 訓練結束時 train.py 會 plt.show()；有桌面環境時會開視窗卡住、下一個專家無法開始。
# 改用不開視窗的後端，只影響畫圖，不影響訓練。
export MPLBACKEND=Agg

EPOCHS=600
IMAGE_SIZE=256        # 已確認：test.py 以 256 重現原 Dice/Ψ1/Ψ5（README 訓練指令亦為 256）
NUM_WORKERS=1         # train.py 預設值（README 範例為 4；只影響資料讀取速度）
BATCH_SIZE=16         # 已確認：best_designed.pth 的優化器更新 4925 次 / 197 epoch = 25 次 = 400 ÷ 16
GRAD_CKPT=0           # batch 16 放得下 4080，不需要
NORM=${PSI_NORM:-wsum} # 新增：batch 內的平均方式（wsum = ÷Σw，原本的結果；mean = ÷張數）
TAG=$([ "$NORM" = wsum ] && echo "" || echo "_$NORM")   # 新增：mean → 輸出到 expert_mean_<proxy>，不覆蓋原結果

for P in psi1_convex psi5_exclusion psi8_rotation; do
  L=$LISTS/wcases_refuge_$P.csv
  [ -f "$L" ] || { echo "找不到清單 $L"; exit 1; }
  R=$OUT/expert${TAG}_$P
  mkdir -p "$R"
  echo "===== $(date '+%F %T') 開始 $P ====="
  python3 train_psi.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --model_selection FunduSegmenter \
    --output_channel 3 --label_n_cls 3 \
    --seed 112316 \
    --train_image_path $D/training/ROIs/images   --train_mask_path $D/training/ROIs/masks \
    --separate_val 1 \
    --val_image_path   $D/validation/ROIs/images --val_mask_path   $D/validation/ROIs/masks \
    --num_workers $NUM_WORKERS \
    --image_size $IMAGE_SIZE \
    --transform_mode designed_transform \
    --results_base_dir "$R" \
    --psi_cases "$L" --psi_factor 3.0 --psi_norm $NORM \
    --grad_ckpt $GRAD_CKPT \
    2>&1 | tee "$R/train.log"
  echo "===== $(date '+%F %T') 完成 $P ====="
done
