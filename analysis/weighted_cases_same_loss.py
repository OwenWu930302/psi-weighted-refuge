"""
被加權病例：同一種 loss（逐張計算）下，加權前後的 Dice — REFUGE
============================================================
格式與 psi-weighted-nnunet/analysis/weighted_cases_same_loss.py 相同（檔名、欄位、彙總方式）。

加權前 = 對照組（train_psi.py 不給清單，權重全為 1，與專家同一條逐張 loss 路徑）
加權後 = Ψ 專家（清單病例 ×3）
兩者只差在「有沒有 ×3 加權」，因此 Δ = 加權本身對該病例的影響。

不與 baseline 比較：baseline 同時是產生加權清單的模型（清單由它的預測挑出），
以它為加權前會混入回歸平均。

預測來源：各模型對 training 400 張的預測（模型訓練時看過該病例 → 反映加權是否讓模型更擬合該病例）
病例 Dice = (OD + OC) / 2；OD = 視盤（類別 1 + 2），OC = 視杯（類別 2），於原圖尺寸計算。

輸入（倉庫內，由 tools/collect_results.sh 複製）：
  results/per_image/control_training.csv
  results/per_image/expert_<proxy>_training.csv
  weight_lists/wcases_refuge_<proxy>.csv

輸出目錄：results/weighted_cases/
  weighted_cases_same_loss_REFUGE.csv      每個被加權病例：加權前 / 後 Dice（病例與 OD、OC）、Δ、提升或下降
  weighted_cases_same_loss_summary.csv     每個專家一列：例數、平均 Dice、Δ、提升 / 下降例數、Wilcoxon p
  weighted_cases_same_loss_REFUGE_all.csv  所有訓練病例（含未加權），欄位 weighted = 是 / 否
  weighted_cases_same_loss_compare.csv     每個專家：加權組 Δ vs 未加權組 Δ，以及兩組差距
                                           （未加權組 = 參考線：扣掉兩個模型整體的偏移，剩下的才是加權本身的效果）

用法（倉庫根目錄）：
  conda activate expertree
  python3 analysis/weighted_cases_same_loss.py               # 專家 loss ÷ Σw（原本的結果）
  python3 analysis/weighted_cases_same_loss.py --norm mean   # 專家 loss ÷ 張數 → results/weighted_cases_mean/
                                                            # （讀 expert_mean_<proxy>_training.csv；對照組相同）
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI = f'{REPO}/results/per_image'
WL = f'{REPO}/weight_lists'
OUT = f'{REPO}/results/weighted_cases'

DS = 'REFUGE'
ORG = ['OD', 'OC']
CTRL = ('refuge_control', 'control_training.csv')
EXPERTS = [('psi1w3', 'expert_psi1_convex_training.csv', 'wcases_refuge_psi1_convex.csv'),
           ('psi5w3', 'expert_psi5_exclusion_training.csv', 'wcases_refuge_psi5_exclusion.csv'),
           ('psi8w3', 'expert_psi8_rotation_training.csv', 'wcases_refuge_psi8_rotation.csv')]
SOURCE = 'training 400 張預測（best epoch）'


def load(fn):
    d = pd.read_csv(f'{PI}/{fn}')
    d['case'] = d['case'].astype(str)
    return d.set_index('case')


def main():
    global OUT, EXPERTS, SOURCE
    ap = argparse.ArgumentParser()                                                          # 新增
    ap.add_argument('--norm', default='wsum', choices=['wsum', 'mean'])                    # 新增
    if ap.parse_args().norm == 'mean':                                                      # 新增：mean 版專家
        OUT += '_mean'
        EXPERTS = [(n + '_mean', fn.replace('expert_', 'expert_mean_'), lf) for n, fn, lf in EXPERTS]
        SOURCE += '；專家 loss ÷ 張數（--psi_norm mean）'
    os.makedirs(OUT, exist_ok=True)
    ctrl = load(CTRL[1])
    rows_w, rows_all, summary, compare = [], [], [], []
    for name, fn, lf in EXPERTS:
        if not os.path.exists(f'{PI}/{fn}'):
            print(f'[略過] {name}：找不到 {fn}')
            continue
        expt = load(fn)
        assert set(expt.index) == set(ctrl.index), f'{name}：病例與對照組不一致'
        wl = set(pd.read_csv(f'{WL}/{lf}')['case'].astype(str))
        miss = wl - set(ctrl.index)
        assert not miss, f'{name}：清單中 {len(miss)} 例不在訓練集'
        for case in sorted(ctrl.index):
            b, a = ctrl.loc[case], expt.loc[case]
            r = dict(expert=name, list_file=lf, case=case, weighted='是' if case in wl else '否',
                     control=CTRL[0], n_folds='single', before_control=b['dice'], after_expert=a['dice'])
            r['delta'] = r['after_expert'] - r['before_control']
            r['result'] = '提升' if r['delta'] > 0 else ('下降' if r['delta'] < 0 else '持平')
            for o in ORG:
                k = f'{o.lower()}_dice'
                r[f'{o}_before'], r[f'{o}_after'] = b[k], a[k]
                r[f'{o}_delta'] = a[k] - b[k]
            rows_all.append(r)
            if case in wl:
                rows_w.append({k: v for k, v in r.items() if k != 'weighted'})

        dw = [r['delta'] for r in rows_all if r['expert'] == name and r['weighted'] == '是']
        du = [r['delta'] for r in rows_all if r['expert'] == name and r['weighted'] == '否']
        bw = [r['before_control'] for r in rows_all if r['expert'] == name and r['weighted'] == '是']
        aw = [r['after_expert'] for r in rows_all if r['expert'] == name and r['weighted'] == '是']
        d = np.array(dw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else np.nan
            pm = float(mannwhitneyu(dw, du).pvalue)
        summary.append(dict(dataset=DS, expert=name, list_file=lf, control=CTRL[0], prediction_source=SOURCE,
                            n_weighted=len(d), before_control=np.mean(bw), after_expert=np.mean(aw),
                            delta=d.mean(), 提升=int((d > 0).sum()), 下降=int((d < 0).sum()), wilcoxon_p=p))
        compare.append(dict(dataset=DS, expert=name, control=CTRL[0],
                            n_weighted=len(dw), delta_weighted=np.mean(dw),
                            加權組_提升=int((np.array(dw) > 0).sum()), 加權組_下降=int((np.array(dw) < 0).sum()),
                            n_unweighted=len(du), delta_unweighted=np.mean(du),
                            未加權組_提升=int((np.array(du) > 0).sum()), 未加權組_下降=int((np.array(du) < 0).sum()),
                            weighted_minus_unweighted=np.mean(dw) - np.mean(du), mannwhitney_p=pm))

    kw = dict(index=False, encoding='utf-8-sig', float_format='%.4f')
    W = pd.DataFrame(rows_w)
    W.to_csv(f'{OUT}/weighted_cases_same_loss_{DS}.csv', **kw)
    pd.DataFrame(rows_all).to_csv(f'{OUT}/weighted_cases_same_loss_{DS}_all.csv', **kw)
    S = pd.DataFrame(summary)
    S.to_csv(f'{OUT}/weighted_cases_same_loss_summary.csv', **kw)
    pd.DataFrame(compare).to_csv(f'{OUT}/weighted_cases_same_loss_compare.csv', **kw)

    pd.set_option('display.width', 200)
    print('【被加權病例：加權前（對照組）→ 加權後（專家）】')
    print(S[['expert', 'n_weighted', 'before_control', 'after_expert', 'delta', '提升', '下降', 'wilcoxon_p']]
          .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n【被加權病例：OD / OC】')
    g = W.groupby('expert', sort=False)[[f'{o}_{s}' for o in ORG for s in ('before', 'after', 'delta')]].mean()
    print(g.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\n已寫入 {OUT}/')


if __name__ == '__main__':
    main()
