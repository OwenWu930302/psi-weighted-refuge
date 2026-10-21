#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFUGE：被加權病例在訓練集上的加權前後 Dice
（格式與 psi-weighted-nnunet/analysis/weighted_cases_same_loss.py 相同）

加權前 = --before baseline：原 baseline（best_designed.pth，也是產生清單的模型）
         --before control ：對照組（train_psi.py 不給清單重跑一次；建議用這個）
         用 baseline 當加權前會混入回歸平均（清單由 baseline 的預測挑出），見 refuge_placebo.py；
         用對照組則專家與對照組都是新的一次訓練，回歸平均兩邊抵消（同 psi-weighted-nnunet 的 ps1ctrl / allctrl）。
加權後 = Ψ 專家（清單病例 ×3）
預測來源 = 模型對 training 400 張的預測（模型訓練時看過這些影像）
         → 反映加權是否讓模型更擬合被加權的病例

輸出（--out 目錄）
  weighted_cases_REFUGE.csv       每個被加權病例：加權前 / 後 Dice（病例、OD、OC）、Δ、提升或下降
  weighted_cases_REFUGE_all.csv   全部 400 例，欄位 weighted = 是 / 否
  weighted_cases_summary.csv      每個專家一列：加權組例數、平均 Dice、Δ、提升 / 下降、Wilcoxon p
  weighted_cases_compare.csv      加權組 Δ vs 未加權組 Δ、兩組差距、Mann-Whitney p
                                  （未加權組 = 參考線：扣掉兩個模型整體的偏移，剩下的才是加權本身的效果）

用法（FunduSegmenter 資料夾內）
  conda run -n expertree python3 refuge_compare.py --root ../psi_expert                    # 對 baseline
  conda run -n expertree python3 refuge_compare.py --root ../psi_expert --before control   # 對對照組
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

EXPERTS = [('psi1_convex', 'wcases_refuge_psi1_convex.csv'),
           ('psi5_exclusion', 'wcases_refuge_psi5_exclusion.csv'),
           ('psi8_rotation', 'wcases_refuge_psi8_rotation.csv')]

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='../psi_expert')
ap.add_argument('--split', default='training')
ap.add_argument('--out', default='')
ap.add_argument('--before', default='baseline', choices=['baseline', 'control'],
                help='加權前的基準：baseline（原模型）或 control（不加權重跑的對照組）')
a = ap.parse_args()
R = a.root
OUT = a.out or (f'{R}/compare_{a.split}' if a.before == 'baseline' else f'{R}/compare_{a.split}_vs_{a.before}')
os.makedirs(OUT, exist_ok=True)

if a.before == 'baseline':
    base_dir = f'{R}/baseline_{a.split}' if a.split == 'training' else f'{R}/{a.split}256'
else:
    base_dir = f'{R}/{a.before}_{a.split}'
B = pd.read_csv(f'{base_dir}/per_image.csv').set_index('case')
print(f'加權前（{a.before}）：{base_dir}/per_image.csv　{len(B)} 例　病例 Dice 平均 {B.dice.mean():.4f}')


def pval(f, *x):
    try:
        return float(f(*x).pvalue)
    except ValueError:
        return np.nan


rows, rows_all, summary, compare = [], [], [], []
for name, lf in EXPERTS:
    ef = f'{R}/expert_{name}_{a.split}/per_image.csv'
    if not os.path.exists(ef):
        print(f'[略過] 找不到 {ef}')
        continue
    E = pd.read_csv(ef).set_index('case')
    cases = sorted(set(pd.read_csv(f'{R}/lists/{lf}')['case'].astype(str)))
    assert set(E.index) == set(B.index), f'{name}：病例與 baseline 不一致'
    w = pd.Series(B.index.isin(cases), index=B.index)

    d = pd.DataFrame({'expert': name, 'case': B.index, 'weighted': np.where(w, '是', '否'),
                      'before': B.dice, 'after': E.dice.reindex(B.index)})
    for s in ['od', 'oc']:
        d[f'{s.upper()}_before'] = B[f'{s}_dice']
        d[f'{s.upper()}_after'] = E[f'{s}_dice'].reindex(B.index)
        d[f'{s.upper()}_delta'] = d[f'{s.upper()}_after'] - d[f'{s.upper()}_before']
    d['delta'] = d.after - d.before
    d['result'] = np.select([d.delta > 0, d.delta < 0], ['提升', '下降'], '不變')
    d = d.reset_index(drop=True)
    rows_all.append(d)
    rows.append(d[d.weighted == '是'].drop(columns='weighted'))

    dw, du = d[d.weighted == '是'], d[d.weighted == '否']
    summary.append(dict(dataset='REFUGE', expert=name, list_file=lf, control=a.before,
                        prediction_source=f'{a.split} 400 張預測', n_weighted=len(dw),
                        before_control=dw.before.mean(), after_expert=dw.after.mean(), delta=dw.delta.mean(),
                        OD_delta=dw.OD_delta.mean(), OC_delta=dw.OC_delta.mean(),
                        提升=int((dw.delta > 0).sum()), 下降=int((dw.delta < 0).sum()),
                        wilcoxon_p=pval(wilcoxon, dw.delta)))
    compare.append(dict(dataset='REFUGE', expert=name,
                        n_weighted=len(dw), delta_weighted=dw.delta.mean(),
                        加權組_提升=int((dw.delta > 0).sum()), 加權組_下降=int((dw.delta < 0).sum()),
                        n_unweighted=len(du), delta_unweighted=du.delta.mean(),
                        未加權組_提升=int((du.delta > 0).sum()), 未加權組_下降=int((du.delta < 0).sum()),
                        weighted_minus_unweighted=dw.delta.mean() - du.delta.mean(),
                        mannwhitney_p=pval(mannwhitneyu, dw.delta, du.delta),
                        overall_before=d.before.mean(), overall_after=d.after.mean(),
                        overall_delta=d.delta.mean(), overall_wilcoxon_p=pval(wilcoxon, d.delta)))

if not summary:
    raise SystemExit('\n→ 一個專家的預測都沒有。請先執行步驟 1：\n'
                     '  python3 refuge_infer_psi.py --split training --image_size 256 --no_psi8 \\\n'
                     '    --ckpt ../psi_expert/expert_<proxy>/saved_weights/best_weights.pth \\\n'
                     '    --out ../psi_expert/expert_<proxy>_training')
kw = dict(index=False, encoding='utf-8-sig', float_format='%.4f')
pd.concat(rows).to_csv(f'{OUT}/weighted_cases_REFUGE.csv', **kw)
pd.concat(rows_all).to_csv(f'{OUT}/weighted_cases_REFUGE_all.csv', **kw)
S, K = pd.DataFrame(summary), pd.DataFrame(compare)
S.to_csv(f'{OUT}/weighted_cases_summary.csv', **kw)
K.to_csv(f'{OUT}/weighted_cases_compare.csv', **kw)

pd.set_option('display.width', 200)
print(f'\n【被加權病例：加權前 / 後（{a.split}）】')
print(f'{"專家":<16}{"n":>4}{"加權前":>9}{"加權後":>9}{"Δ":>9}{"ΔOD":>9}{"ΔOC":>9}{"升/降":>9}{"p":>10}')
for r in summary:
    print(f'{r["expert"]:<16}{r["n_weighted"]:>4}{r["before_control"]:>9.4f}{r["after_expert"]:>9.4f}'
          f'{r["delta"]:>+9.4f}{r["OD_delta"]:>+9.4f}{r["OC_delta"]:>+9.4f}'
          f'{str(r["提升"]) + "/" + str(r["下降"]):>9}{r["wilcoxon_p"]:>10.2e}')
print(f'\n【加權組 vs 未加權組（{a.split}）】')
print(f'{"專家":<16}{"加權組Δ":>9}{"升/降":>9}{"未加權組Δ":>10}{"升/降":>9}{"差距":>9}{"MW p":>10}{"整體Δ":>9}{"整體p":>10}')
for r in compare:
    print(f'{r["expert"]:<16}{r["delta_weighted"]:>+9.4f}'
          f'{str(r["加權組_提升"]) + "/" + str(r["加權組_下降"]):>9}'
          f'{r["delta_unweighted"]:>+10.4f}'
          f'{str(r["未加權組_提升"]) + "/" + str(r["未加權組_下降"]):>9}'
          f'{r["weighted_minus_unweighted"]:>+9.4f}{r["mannwhitney_p"]:>10.2e}'
          f'{r["overall_delta"]:>+9.4f}{r["overall_wilcoxon_p"]:>10.2e}')
print(f'\n已寫入 {OUT}/')
