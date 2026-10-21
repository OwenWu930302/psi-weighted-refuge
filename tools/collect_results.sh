#!/bin/bash
# ============================================================
# 把本機的實驗結果複製進倉庫（只複製 CSV 與摘要，不含影像、預測圖、權重）
#
# 用法（在倉庫根目錄）：
#   bash tools/collect_results.sh
#   PSI_EXPERT=<psi_expert 路徑> FS=<FunduSegmenter 路徑> bash tools/collect_results.sh
# 可重複執行：已存在的檔案會被覆蓋；找不到的來源會列出「略過」。
# ============================================================
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PE="${PSI_EXPERT:-$HOME/桌面/REFUGE/psi_expert}"
FS="${FS:-$HOME/桌面/REFUGE/FunduSegmenter}"
PE="$(cd "$PE" 2>/dev/null && pwd || echo "$PE")"   # 轉成絕對路徑（下面會 cd 到倉庫）
FS="$(cd "$FS" 2>/dev/null && pwd || echo "$FS")"
TOP="$(dirname "$FS")"                       # 舊版 Ψ 結果（caselevel_refuge.csv 等）所在
cd "$REPO"
echo "倉庫        $REPO"
echo "psi_expert  $PE"
echo "FunduSeg.   $FS"
n_ok=0; n_skip=0
cp_() {  # cp_ 來源 目的
  if [ -f "$1" ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; echo "  ✓ $2"; n_ok=$((n_ok+1))
  else echo "  - 略過（找不到 $1）"; n_skip=$((n_skip+1)); fi
}

echo "【加權清單】"
for P in psi1_convex psi5_exclusion psi8_rotation; do
  cp_ "$PE/lists/wcases_refuge_$P.csv" "weight_lists/wcases_refuge_$P.csv"
done

echo "【每張影像的 Dice / Ψ】"
cp_ "$PE/baseline_training/per_image.csv" results/per_image/baseline_training.csv
cp_ "$PE/validation256/per_image.csv"     results/per_image/baseline_validation.csv
cp_ "$PE/testing256/per_image.csv"        results/per_image/baseline_testing.csv
for P in psi1_convex psi5_exclusion psi8_rotation; do
  cp_ "$PE/expert_${P}_training/per_image.csv" "results/per_image/expert_${P}_training.csv"
  cp_ "$PE/expert_${P}_testing/per_image.csv"  "results/per_image/expert_${P}_testing.csv"
done
cp_ "$PE/control_training/per_image.csv" results/per_image/control_training.csv
cp_ "$PE/control_testing/per_image.csv"  results/per_image/control_testing.csv

echo "【proxy 篩選：病例層級 Ψ 表】"
cp_ "$PE/validation256/caselevel_refuge_validation256.csv"  results/proxy/caselevel_refuge_validation.csv
cp_ "$TOP/caselevel_refuge.csv"                             results/proxy/caselevel_refuge_testing_original.csv
cp_ "$PE/testing256/caselevel_refuge_psi8at256.csv"         results/proxy/caselevel_refuge_testing_psi8at256.csv

echo "【加權前後比較 / 安慰劑】"
for d in compare_training compare_training_vs_control compare_testing compare_testing_vs_control; do
  if [ -d "$PE/$d" ]; then
    for f in "$PE/$d"/*.csv; do cp_ "$f" "results/$d/$(basename "$f")"; done
  else echo "  - 略過（找不到 $PE/$d）"; n_skip=$((n_skip+1)); fi
done

echo "【訓練曲線（從 log 擷取每個 epoch 的 loss / validation Dice）】"
python3 - "$FS/repro_notes/train_designed.log" results/curves/baseline.csv \
          "$PE/expert_psi1_convex/train.log"    results/curves/expert_psi1_convex.csv \
          "$PE/expert_psi5_exclusion/train.log" results/curves/expert_psi5_exclusion.csv \
          "$PE/expert_psi8_rotation/train.log"  results/curves/expert_psi8_rotation.csv \
          "$PE/control/train.log"               results/curves/control.csv <<'PY'
import sys, re, csv, os
args = sys.argv[1:]
pat = {'train_loss': r'training loss', 'val_od': r'validation OD Dice', 'val_oc': r'validation OC Dice',
       'val_mean': r'validation mean Dice', 'weighted_samples': r'weighted samples'}
for src, dst in zip(args[::2], args[1::2]):
    if not os.path.exists(src):
        print(f'  - 略過（找不到 {src}）'); continue
    txt = open(src, encoding='utf-8', errors='ignore').read().replace('\r', '\n')
    rows = {}
    for k, p in pat.items():
        for ep, v in re.findall(rf'Epoch_(\d+) {p}: ([-0-9.eE]+)', txt):
            rows.setdefault(int(ep), {})[k] = v
    if not rows:
        print(f'  - 略過（{src} 沒有 epoch 紀錄）'); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch'] + list(pat))
        for ep in sorted(rows):
            w.writerow([ep] + [rows[ep].get(k, '') for k in pat])
    best = max(rows, key=lambda e: float(rows[e].get('val_mean', '-1')))
    print(f'  ✓ {dst}（{len(rows)} 個 epoch，最佳 epoch {best}，val mean {float(rows[best]["val_mean"]):.4f}）')
PY

echo "【訓練摘要（log 中的關鍵行）】"
mkdir -p results/logs
for L in "$FS/repro_notes/train_designed.log:baseline" "$PE/expert_psi1_convex/train.log:expert_psi1_convex" \
         "$PE/expert_psi5_exclusion/train.log:expert_psi5_exclusion" "$PE/expert_psi8_rotation/train.log:expert_psi8_rotation" \
         "$PE/control/train.log:control"; do
  src="${L%%:*}"; name="${L##*:}"
  if [ -f "$src" ]; then
    grep -aE "Number of|Ψ加權|Start training|best weights|Training time" "$src" | tr '\r' '\n' \
      | grep -aE "Number of|Ψ加權|Start training|best weights|Training time" > "results/logs/${name}_summary.txt"
    echo "  ✓ results/logs/${name}_summary.txt"
  fi
done

echo
echo "完成：複製 $n_ok 個檔案，略過 $n_skip 個（尚未產生的結果會在之後重跑本腳本時補上）"
echo "接著： git add -A && git status   確認內容後再 commit"
