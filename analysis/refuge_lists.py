#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFUGE 加權清單：一致性檢查 + 產生清單

check  用本程式重算的 testing 結果，比對原本的
       caselevel_refuge.csv（Ψ1 / Ψ5 / Ψ8 / dice）與 per_image_dice_designed.csv（OD / OC）
       → 確認 refuge_infer_psi.py 與原流程一致，才能信任訓練集的數值
proxy  以原 caselevel 為底、只替換 Ψ8，呼叫 recompute_caselevel.report() 重新檢定
make   依 make_weight_lists.py 的 select() 規則（逐字複製）產生三份清單：
         Ψ1 top30、Ψ5 bottom30（反向）、Ψ8 top30；並列者全部納入

用法
  python3 refuge_lists.py check --new ../psi_expert/check_testing/per_image.csv \
          --old_case ../caselevel_refuge.csv --old_dice repro_notes/per_image_dice_designed.csv
  python3 refuge_lists.py make  --table ../psi_expert/baseline_training/per_image.csv \
          --out ../psi_expert/lists
"""
import os, sys, argparse
import numpy as np
import pandas as pd

SPECS = [  # (欄位, 規則, 清單檔名)
    ('psi1_convex',    'top30',    'wcases_refuge_psi1_convex.csv'),
    ('psi5_exclusion', 'bottom30', 'wcases_refuge_psi5_exclusion.csv'),
    ('psi8_rotation',  'top30',    'wcases_refuge_psi8_rotation.csv'),
]


def select(d, col, rule):
    """逐字複製自 psi-weighted-nnunet/psi/make_weight_lists.py"""
    k = int(round(len(d) * 0.30))
    if rule == 'top30':
        thr = d[col].sort_values(ascending=False).iloc[k - 1]
        return d[d[col] >= thr], f'>= {thr:.6g}'
    if rule == 'bottom30':
        thr = d[col].sort_values(ascending=True).iloc[k - 1]
        return d[d[col] <= thr], f'<= {thr:.6g}'
    if rule == 'gt0':
        return d[d[col] > 0], '> 0'
    raise ValueError(rule)


def cmp(a, b, name, tol):
    m = a.notna() & b.notna()
    diff = (a[m] - b[m]).abs()
    nan_mis = int((a.isna() != b.isna()).sum())
    bad = int((diff > tol).sum())
    flag = '✓' if bad == 0 and nan_mis == 0 else '✗'
    print(f'  {flag} {name:<16} n={int(m.sum()):>3}  最大差 {diff.max():.2e}  '
          f'超過 {tol:g} 的 {bad} 張  NaN 不一致 {nan_mis} 張')
    if bad:
        top = diff.sort_values(ascending=False).head(5)
        print('      差最多：' + ', '.join(f'{i}({v:.3g})' for i, v in top.items()))
    return bad == 0 and nan_mis == 0


def do_check(a):
    new = pd.read_csv(a.new).set_index('case')
    oc = pd.read_csv(a.old_case).set_index('case')
    print(f'新 {len(new)} 例  舊 caselevel {len(oc)} 例  共同 {len(new.index.intersection(oc.index))} 例')
    ok = True
    print('【Ψ 與病例 Dice vs caselevel_refuge.csv】')
    for c in ['dice', 'psi1_convex', 'psi5_exclusion', 'psi2_shadow', 'psi8_rotation']:
        if c in new and c in oc:
            # Dice 與 Ψ1/Ψ5 為 fp32 推論，應幾乎相同；Ψ2/Ψ8 為 AMP，容許稍大
            ok &= cmp(new[c], oc[c].reindex(new.index), c, 2e-2 if c in ('psi2_shadow', 'psi8_rotation') else 1e-3)
    if a.old_dice and os.path.exists(a.old_dice):
        od = pd.read_csv(a.old_dice)
        od['case'] = od['filename'].str.replace(r'\.\w+$', '', regex=True)
        od = od.set_index('case')
        print('【OD / OC Dice vs per_image_dice_designed.csv】')
        ok &= cmp(new['od_dice'], od['od_dice'].reindex(new.index), 'od_dice', 1e-3)
        ok &= cmp(new['oc_dice'], od['oc_dice'].reindex(new.index), 'oc_dice', 1e-3)
    # 排名一致性比數值更重要：清單取的是排名
    from scipy.stats import spearmanr
    print('【排名一致性（Spearman）】')
    for c in ['psi1_convex', 'psi5_exclusion', 'psi2_shadow', 'psi8_rotation']:
        if c in new and c in oc:
            x, y = new[c], oc[c].reindex(new.index)
            m = x.notna() & y.notna()
            print(f'  {c:<16} ρ = {spearmanr(x[m], y[m]).correlation:.6f}')
    print('\n→ 全部 ✓：可以用同一支程式產生訓練集的 Ψ' if ok else
          '\n→ 有 ✗：先不要產生清單，把以上輸出貼回來')


def do_proxy(a):
    """以原 caselevel 為底，只把 Ψ8 換成新算的值，用 recompute_caselevel.report() 重新檢定。"""
    sys.path.insert(0, os.path.join(os.path.expanduser(a.repo), 'psi'))
    from recompute_caselevel import report
    new = pd.read_csv(a.new).set_index('case')
    old = pd.read_csv(a.old_case)
    if len(new) != len(old):
        sys.exit(f'新表 {len(new)} 例、舊表 {len(old)} 例：請用完整 400 張 testing')
    print('【先確認 Dice / Ψ1 / Ψ5 與原表完全相同（全部 400 張）】')
    o = old.set_index('case')
    ok = True
    for c in ['dice', 'psi1_convex', 'psi5_exclusion']:
        ok &= cmp(new[c], o[c].reindex(new.index), c, 1e-3)
    if not ok:
        sys.exit('→ 有 ✗，停止')
    thr = old.dice.quantile(0.30)
    y = (old.dice <= thr).astype(int).values
    lab = f'相對口徑：最低 30%（Dice ≤ {thr:.4f}，含並列）'
    report('REFUGE（原表：Ψ8 @224）— 應與你的表相同', old, y, lab)
    cd = old.copy()
    cd['psi8_rotation'] = cd['case'].map(new['psi8_rotation'])
    report(f'REFUGE（Ψ8 @{a.psi8_size}）— 只有 Ψ8 這一列會變', cd, y, lab)
    out = os.path.join(os.path.dirname(a.new), f'caselevel_refuge_psi8at{a.psi8_size}.csv')
    cd.to_csv(out, index=False)
    print(f'\n已寫入 {out}')


def do_make(a):
    d = pd.read_csv(a.table)
    d['case'] = d['case'].astype(str)
    os.makedirs(a.out, exist_ok=True)
    print(f'來源 {a.table}（{len(d)} 例，病例 Dice 平均 {d.dice.mean():.4f}）')
    print(f'{"proxy":<16} {"規則":<9} {"門檻":<14} {"例數":>4} {"清單 Dice":>9} {"其餘 Dice":>9}  檔案')
    sets = {}
    for col, rule, fn in SPECS:
        if col not in d or d[col].isna().all():
            print(f'{col:<16} 欄位不存在或全為 NaN，略過'); continue
        n_nan = int(d[col].isna().sum())
        dd = d[d[col].notna()]
        s, thr = select(dd, col, rule)
        rest = d[~d.case.isin(s.case)]
        s[['case', col, 'dice']].sort_values(col, ascending=(rule == 'bottom30')) \
            .to_csv(os.path.join(a.out, fn), index=False)
        sets[col] = set(s.case)
        print(f'{col:<16} {rule:<9} {thr:<14} {len(s):>4} {s.dice.mean():>9.4f} {rest.dice.mean():>9.4f}  {fn}'
              + (f'   （{n_nan} 例 NaN 不納入排序）' if n_nan else ''))
    ks = list(sets)
    if len(ks) > 1:
        print('清單重疊：' + '  '.join(f'{ks[i]}∩{ks[j]} = {len(sets[ks[i]] & sets[ks[j]])}'
                                    for i in range(len(ks)) for j in range(i + 1, len(ks))))
    print(f'已寫入 {a.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('check')
    c.add_argument('--new', required=True)
    c.add_argument('--old_case', required=True)
    c.add_argument('--old_dice', default='')
    x = sp.add_parser('proxy')
    x.add_argument('--new', required=True)
    x.add_argument('--old_case', required=True)
    x.add_argument('--repo', default='~/psi_workspace/psi-weighted-nnunet')
    x.add_argument('--psi8_size', default='256')
    m = sp.add_parser('make')
    m.add_argument('--table', required=True)
    m.add_argument('--out', required=True)
    a = ap.parse_args()
    {'check': do_check, 'proxy': do_proxy, 'make': do_make}[a.cmd](a)
