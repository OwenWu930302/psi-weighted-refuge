#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驗證 train_psi.py 的 --grad_ckpt 1（encoder gradient checkpointing）不改變 loss 與梯度，
並估計 batch 32 時的 GPU 記憶體。

做法：同一個模型、同一批影像、同一個亂數種子，先不用 checkpointing 算一次 loss.backward()，
再啟用 checkpointing 算一次，逐參數比對所有可學習參數的梯度。
模型凍結方式與 train.py 相同（encoder 凍結、只有 adapter / pre / decoder / post 可學習）。

用法（FunduSegmenter 資料夾內）：
  conda activate fundusegmenter
  python3 check_grad_ckpt.py
"""
import os, sys, argparse, copy
import numpy as np
import monai
import torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.transform import no_transform
from refuge_infer_psi import build_model
from train_psi import enable_grad_ckpt

ap = argparse.ArgumentParser()
ap.add_argument('--data_root', default='/mnt/data2t/datasets/REFUGE/REFUGE')
ap.add_argument('--ckpt', default='repro_notes/best_designed.pth')
ap.add_argument('--image_size', type=int, default=256)
ap.add_argument('--batches', default='4,8')
a = ap.parse_args()
dev = torch.device('cuda')

# 資料：前 8 張 training ROI（不做隨機增強，確保兩次輸入相同）
ri, rm = f'{a.data_root}/training/ROIs/images', f'{a.data_root}/training/ROIs/masks'
fi, fm = sorted(os.listdir(ri))[:8], sorted(os.listdir(rm))[:8]
t_img, t_both = no_transform(a.image_size)
X, Y = [], []
for i, m in zip(fi, fm):
    img = np.asarray(Image.open(f'{ri}/{i}'))
    msk = np.asarray(Image.open(f'{rm}/{m}'))
    msk = (msk[:, :, 0] if msk.ndim == 3 else msk).astype(np.uint8)
    msk[msk == 0] = 2; msk[msk == 128] = 1; msk[msk == 255] = 0
    r = t_both(image=t_img(image=img)['image'], mask=msk)
    X.append(r['image']); Y.append(r['mask'].long())
X, Y = torch.stack(X), torch.stack(Y)

# 模型：與 train.py 相同的凍結方式
base = build_model()
ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
base.load_state_dict(ck['model'], strict=True)
del ck
for p in base.encoder.parameters():
    p.requires_grad = False
for blk in base.encoder.blocks:
    for p in blk.adapter.parameters():
        p.requires_grad = True

dice_loss = monai.losses.DiceLoss(include_background=True, squared_pred=False, softmax=True)
ce_loss = torch.nn.CrossEntropyLoss()


def run(model, B):
    """一次 forward + backward，回傳 loss、各參數梯度、峰值記憶體（GB）。"""
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    x = X[:B].to(dev)
    y = F.one_hot(Y[:B].to(dev), num_classes=3).permute(0, 3, 1, 2).float()
    out = model(x)
    loss = dice_loss(out, y) + ce_loss(out, y)
    loss.backward()
    g = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    return loss.item(), g, torch.cuda.max_memory_allocated() / 2**30


peaks = {False: {}, True: {}}
ok_all = True
for B in [int(b) for b in a.batches.split(',')]:
    m0 = copy.deepcopy(base).to(dev)
    l0, g0, p0 = run(m0, B)
    m1 = copy.deepcopy(base).to(dev)
    # 兩個副本的 BN running stats 起點相同；train 模式的輸出不依賴 running stats
    enable_grad_ckpt(m1)
    l1, g1, p1 = run(m1, B)
    peaks[False][B], peaks[True][B] = p0, p1
    dmax = max((g0[n] - g1[n]).abs().max().item() for n in g0)
    gmax = max(g0[n].abs().max().item() for n in g0)
    same_keys = list(m0.state_dict().keys()) == list(m1.state_dict().keys())
    ok = abs(l0 - l1) < 1e-6 and dmax <= 1e-5 * max(1.0, gmax) and same_keys
    ok_all &= ok
    print(f'batch {B}：loss {l0:.8f} vs {l1:.8f}（差 {abs(l0-l1):.1e}）  '
          f'梯度最大差 {dmax:.1e}（梯度最大值 {gmax:.1e}）  state_dict 鍵名相同 {same_keys}  '
          f'{"✓" if ok else "✗"}')
    print(f'          GPU 峰值：不用 {p0:.2f} GB → 用 checkpointing {p1:.2f} GB')
    del m0, m1, g0, g1
    torch.cuda.empty_cache()

bs = sorted(peaks[False])
if len(bs) >= 2:
    b1, b2 = bs[0], bs[-1]
    for k, name in [(False, '不用'), (True, '用 checkpointing')]:
        slope = (peaks[k][b2] - peaks[k][b1]) / (b2 - b1)
        est = peaks[k][b2] + slope * (32 - b2)
        print(f'batch 32 估計（{name}）：約 {est:.1f} GB（4080 可用約 15.5 GB）')

print('\n→ 全部 ✓：checkpointing 不改變 loss 與梯度' if ok_all else '\n→ 有 ✗：不要使用 --grad_ckpt，把輸出貼回來')
