#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFUGE proxy 篩選（任意 split，例如 validation）：完全沿用倉庫的程式

  Ψ 函式      ← import 自 psi/run_all_psi.py（psi1_convex … psi10_centroid）
  遮罩定義    ← 與 run_all_psi.load_refuge() 相同的迴圈（cup = 0、disc = 0|128、rim = 128）
  Ψ8          ← refuge_infer_psi.py 算好的 per_image.csv（256，旋轉 15°），以 case 合併到兩個結構
               （與 recompute_caselevel.load('refuge') 合併 psi2_psi8_refuge.csv 的方式相同）
  Ψ2          ← per_image.csv 的 psi2_shadow（refuge_infer_psi.py 算的）；或以 --psi2_csv 指定
  病例層級    ← import recompute_caselevel.to_case（dice 取平均、Ψ 取 nanmax）
  檢定        ← import recompute_caselevel.report（對照口徑 Dice < 0.80、最低 20% / 30%）

用法（FunduSegmenter 資料夾內）：
  conda run -n expertree python3 refuge_select_proxy.py \
      --pred ../psi_expert/validation256 --repo ~/桌面/論文/psi-weighted-nnunet
"""
import os, sys, glob, argparse
import numpy as np
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True, help='refuge_infer_psi.py 的輸出資料夾（含 per_image.csv 與 map/）')
ap.add_argument('--repo', required=True, help='psi-weighted-nnunet 倉庫位置')
ap.add_argument('--psi2_csv', default='', help='（選用）含 case, psi2_shadow 欄位的 csv')
a = ap.parse_args()
repo = os.path.expanduser(a.repo)
sys.path.insert(0, os.path.join(repo, 'psi'))
import run_all_psi as P                                  # noqa: E402  倉庫的 Ψ 函式
from recompute_caselevel import to_case, report, QUANTILES, DELTA   # noqa: E402

per = pd.read_csv(os.path.join(a.pred, 'per_image.csv'))
per['case'] = per['case'].astype(str)
dice = {r.case: {'OD': r.od_dice, 'OC': r.oc_dice} for r in per.itertuples()}
maps = sorted(glob.glob(os.path.join(a.pred, 'map', '*.png')))
if len(maps) != len(per):
    sys.exit(f'預測圖 {len(maps)} 張與 per_image.csv {len(per)} 例不符（推論時不要加 --no_maps）')

# ---- 與 run_all_psi.load_refuge() 相同的迴圈 ----
rows = []
for f in maps:
    n = os.path.basename(f)[:-4]
    if n not in dice:
        continue
    arr = np.asarray(Image.open(f))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    cup = (arr == 0)
    disc = (arr == 0) | (arr == 128)
    rim = (arr == 128)
    yield_masks = {'OC': cup, 'OD': disc}
    diag = np.sqrt(sum(s ** 2 for s in arr.shape))
    for name, m in yield_masks.items():
        rows.append(dict(
            case=n, structure=name,
            psi1_convex=P.psi1_convex(m),
            psi2_shadow=P.psi2_shadow(),
            psi3_inclusion=P.psi3_inclusion(cup, disc) if name == 'OC' else np.nan,
            psi4_adjacency=P.psi4_adjacency(cup, rim),
            psi5_exclusion=P.psi5_exclusion(m, [rim] if name == 'OC' else [cup]),
            psi6_cc=P.psi6_cc(m),
            psi7_genus=P.psi7_genus(rim),
            psi8_rotation=P.psi8_rotation(),
            psi9_nesting=P.psi9_nesting(cup, disc) if name == 'OC' else np.nan,
            psi10_centroid=P.psi10_centroid(m, disc, diag),
            voxels=int(m.sum()),
            dice=dice[n][name], Z=int(dice[n][name] < P.DELTA)))
d = pd.DataFrame(rows)

# ---- 合併 Ψ8（與 recompute_caselevel 合併 psi2_psi8_refuge.csv 相同：以 case 合併到兩個結構）----
if 'psi8_rotation' in per.columns:
    d = d.drop(columns=['psi8_rotation']).merge(per[['case', 'psi8_rotation']], on='case', how='left')
    print('  [合併] Ψ8（per_image.csv，256）')
if a.psi2_csv:
    e = pd.read_csv(a.psi2_csv); e['case'] = e['case'].astype(str)
    d = d.drop(columns=['psi2_shadow']).merge(e[['case', 'psi2_shadow']], on='case', how='left')
    print(f'  [合併] Ψ2（{a.psi2_csv}）')
elif 'psi2_shadow' in per.columns:
    d = d.drop(columns=['psi2_shadow']).merge(per[['case', 'psi2_shadow']], on='case', how='left')
    print('  [合併] Ψ2（per_image.csv）')
else:
    print('  [未合併] Ψ2 為 NaN（per_image.csv 沒有 Ψ2，也未提供 --psi2_csv）')

# ---- 病例層級與檢定：倉庫的 to_case / report ----
cd = to_case(d)
cd = cd[~cd.dice.isna()].reset_index(drop=True)
split = os.path.basename(os.path.normpath(a.pred))
out = os.path.join(a.pred, f'caselevel_refuge_{split}.csv')
cd.to_csv(out, index=False)
print(f'  病例層級：{len(d)} 筆結構 → {len(cd)} 個病例；病例 Dice min={cd.dice.min():.4f} '
      f'max={cd.dice.max():.4f} mean={cd.dice.mean():.4f}')

y_abs = (cd.dice < DELTA).astype(int).values
if y_abs.sum() > 0:
    report(f'REFUGE（{split}）', cd, y_abs, f'對照口徑：Dice < {DELTA}')
else:
    print(f'\n[對照口徑 Dice < {DELTA}] 零低分樣本，無法檢定')
for q in QUANTILES:
    thr = cd.dice.quantile(q)
    y = (cd.dice <= thr).astype(int).values
    report(f'REFUGE（{split}）', cd, y, f'相對口徑：最低 {int(q*100)}%（Dice ≤ {thr:.4f}，含並列）')
print(f'\n已寫入 {out}')
