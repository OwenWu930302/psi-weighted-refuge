#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFUGE 加權效果的安慰劑檢查：「被加權病例掉得少」是加權造成的，還是回歸平均 / 病例本身的特性？

只讀現有結果，不需 GPU：
  baseline_training/per_image.csv、expert_<proxy>_training/per_image.csv、lists/wcases_refuge_<proxy>.csv
Δ = 專家 Dice − 加權前 Dice（同一張訓練影像；加權前 = --before baseline 或 control）

【檢查一：交叉清單安慰劑】
  在模型 M 裡，計算清單 L 的「差距」= L 的病例平均 Δ − 其他病例平均 Δ
    L 是 M 自己的清單（對角線）→ 真正被加權的效果
    L 是別的清單（非對角線）   → 只用「在 M 裡沒被加權」的病例：
                                  L 中不屬於 M 清單的病例  vs  兩份清單都不屬於的病例
                                  這些病例在 M 裡都沒有加權，差距若仍很大 → 不是加權造成的
【檢查二：控制 baseline Dice 後的差距（排除回歸平均）】
  依 baseline Dice 把 400 例分成 5 組（每組 80 例），只在同一組內比較加權 vs 未加權，再加權平均。
  p 值：在每組內隨機打亂「是否加權」10000 次（分層置換檢定）。
  另以線性迴歸 Δ ~ 加權 + baseline Dice 交叉確認。
【檢查三：回歸平均是否存在】
  未加權病例中，Δ 對 baseline Dice 的斜率；斜率為負 = baseline 越低的病例 Δ 越高 = 有回歸平均。

用法（FunduSegmenter 資料夾內）：
  conda run -n expertree python3 refuge_placebo.py --root ../psi_expert
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy import stats

EXPERTS = ['psi1_convex', 'psi5_exclusion', 'psi8_rotation']
SHORT = {'psi1_convex': 'Ψ1', 'psi5_exclusion': 'Ψ5', 'psi8_rotation': 'Ψ8'}

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='../psi_expert')
ap.add_argument('--n_perm', type=int, default=10000)
ap.add_argument('--n_bins', type=int, default=5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--before', default='baseline', choices=['baseline', 'control'])
a = ap.parse_args()
R = a.root
rng = np.random.default_rng(a.seed)

B = pd.read_csv(f'{R}/{a.before}_training/per_image.csv').set_index('case')
print(f'加權前：{a.before}_training/per_image.csv')
before = B['dice']
cases = before.index
bins = pd.qcut(before.rank(method='first'), a.n_bins, labels=False).values   # baseline Dice 五分位
lists = {p: set(pd.read_csv(f'{R}/lists/wcases_refuge_{p}.csv')['case'].astype(str)) for p in EXPERTS}
delta = {}
for p in EXPERTS:
    E = pd.read_csv(f'{R}/expert_{p}_training/per_image.csv').set_index('case')
    delta[p] = (E['dice'].reindex(cases) - before).values
inL = {p: cases.isin(lists[p]) for p in EXPERTS}


def strat_gap(d, g, use):
    """分層（baseline Dice 五分位）差距：各組內 (g 組平均 − 非 g 組平均)，以 g 組例數加權平均。"""
    num = den = 0.0
    for b in range(a.n_bins):
        m = use & (bins == b)
        n1, n0 = (m & g).sum(), (m & ~g).sum()
        if n1 and n0:
            num += n1 * (d[m & g].mean() - d[m & ~g].mean())
            den += n1
    return num / den if den else np.nan


def strat_perm_p(d, g, use, obs):
    """分層置換檢定（雙尾）：各組內打亂 g 的標籤。"""
    idx = [np.where(use & (bins == b))[0] for b in range(a.n_bins)]
    cnt = 0
    gg = g.copy()
    for _ in range(a.n_perm):
        for ix in idx:
            gg[ix] = rng.permutation(g[ix])
        if abs(strat_gap(d, gg, use)) >= abs(obs) - 1e-15:
            cnt += 1
    return (cnt + 1) / (a.n_perm + 1)


def ols(d, g, x, use):
    """Δ ~ 1 + g + (x − mean)，回傳 g 的係數與 p 值。"""
    y = d[use]
    X = np.column_stack([np.ones(use.sum()), g[use].astype(float), x[use] - x[use].mean()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta[1] / se[1]
    return beta[1], 2 * stats.t.sf(abs(t), dof)


xb = before.values
rows = []
for M in EXPERTS:
    d = delta[M]
    for L in EXPERTS:
        if L == M:
            use = np.ones(len(cases), bool)
            g = inL[L]
            kind = '真正加權'
        else:
            use = ~inL[M]                       # 只用在 M 裡沒被加權的病例
            g = inL[L] & use
            kind = '安慰劑'
        raw = d[use & g].mean() - d[use & ~g].mean()
        adj = strat_gap(d, g, use)
        p_adj = strat_perm_p(d, g, use, adj)
        b_ols, p_ols = ols(d, g, xb, use)
        rows.append(dict(model=M, list=L, kind=kind, n_list=int((use & g).sum()), n_rest=int((use & ~g).sum()),
                         gap_raw=raw, p_raw=stats.mannwhitneyu(d[use & g], d[use & ~g]).pvalue,
                         gap_adj=adj, p_adj=p_adj, gap_ols=b_ols, p_ols=p_ols,
                         base_dice_list=xb[use & g].mean(), base_dice_rest=xb[use & ~g].mean()))
T = pd.DataFrame(rows)
out = f'{R}/compare_training' if a.before == 'baseline' else f'{R}/compare_training_vs_{a.before}'
os.makedirs(out, exist_ok=True)
T.to_csv(f'{out}/placebo_matrix.csv', index=False, encoding='utf-8-sig', float_format='%.6f')


def star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


print('【檢查一＋二：差距矩陣】列 = 模型，欄 = 清單；對角線 = 真正被加權，其餘 = 安慰劑（在該模型中沒被加權）')
print('  原始差距 = 清單病例平均 Δ − 其他病例平均 Δ')
print(f'  控制後差距 = 只在加權前（{a.before}）Dice 相近的病例之間比較（排除回歸平均）\n')
for col, pcol, title in [('gap_raw', 'p_raw', '原始差距'), ('gap_adj', 'p_adj', f'控制加權前（{a.before}）Dice 後的差距')]:
    print(f'  {title}')
    print(f'  {"模型 ＼ 清單":<14}' + ''.join(f'{SHORT[L] + " 清單":>20}' for L in EXPERTS))
    for M in EXPERTS:
        cells = []
        for L in EXPERTS:
            r = T[(T.model == M) & (T.list == L)].iloc[0]
            mark = '◆' if L == M else ' '
            cells.append(f'{mark}{r[col]:+.4f}{star(r[pcol]):<3}(n={r.n_list:>3})')
        print(f'  {SHORT[M] + " 專家":<14}' + ''.join(f'{c:>20}' for c in cells))
    print()
print('  ◆ = 真正被加權　* p<0.05　** p<0.01　*** p<0.001（控制後為分層置換檢定）')

print(f'\n【線性迴歸交叉確認】Δ ~ 是否在清單 + 加權前（{a.before}）Dice')
print(f'  {"模型":<8}{"清單":<8}{"類型":<8}{"係數":>10}{"p":>11}')
for _, r in T.iterrows():
    print(f'  {SHORT[r.model]:<8}{SHORT[r.list]:<8}{r.kind:<8}{r.gap_ols:>+10.4f}{r.p_ols:>11.2e}')

print(f'\n【檢查三：回歸平均是否存在】未加權病例中 Δ 對加權前（{a.before}）Dice 的斜率（負 = 有回歸平均）')
for M in EXPERTS:
    u = ~inL[M]
    s = stats.linregress(xb[u], delta[M][u])
    print(f'  {SHORT[M]} 專家：斜率 {s.slope:+.3f}　r = {s.rvalue:+.3f}　p = {s.pvalue:.2e}　（n = {u.sum()}）')

print(f'\n已寫入 {out}/placebo_matrix.csv')
