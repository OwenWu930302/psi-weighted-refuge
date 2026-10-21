#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出 baseline 訓練時的 --image_size：
用 best_designed.pth（epoch 197）照 train.py 的 validation 流程逐行重算 validation mean Dice，
與 train_designed.log 記錄的 epoch 197 數值比對。能重現的尺寸就是訓練時的尺寸。

train.py 的 validation 流程（照抄）：
  validation_transform(image_size) → model → bicubic 插值到 mask 尺寸（ROI 800×800）
  → argmax(softmax) → one-hot；OD = 類別 1+2，OC = 類別 2
  monai DiceMetric 逐張累加，每張之後 append 一次 aggregate()（累積平均），
  epoch 值 = 這些累積平均的平均；mean Dice = (OD + OC) / 2

用法（FunduSegmenter 資料夾內）：
  conda activate fundusegmenter
  python3 check_train_size.py
"""
import os, sys, argparse, re
import numpy as np
import monai
import torch
from torch.nn import functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.transform import validation_transform
from refuge_infer_psi import build_model

ap = argparse.ArgumentParser()
ap.add_argument('--data_root', default='/mnt/data2t/datasets/REFUGE/REFUGE')
ap.add_argument('--ckpt', default='repro_notes/best_designed.pth')
ap.add_argument('--log', default='repro_notes/train_designed.log')
ap.add_argument('--sizes', default='224,256')
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
ep = ck['epoch']
txt = open(a.log, encoding='utf-8', errors='ignore').read()
rec = {k: float(re.search(rf'Epoch_{ep} validation {k} Dice: ([0-9.]+)', txt).group(1))
       for k in ['OD', 'OC', 'mean']}
print(f'log 記錄的 epoch {ep}：OD {rec["OD"]:.10f}  OC {rec["OC"]:.10f}  mean {rec["mean"]:.10f}')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model()
model.load_state_dict(ck['model'], strict=True)
del ck
model.to(dev).eval()

vi, vm = f'{a.data_root}/validation/ROIs/images', f'{a.data_root}/validation/ROIs/masks'
imgs, msks = sorted(os.listdir(vi)), sorted(os.listdir(vm))

for size in [int(s) for s in a.sizes.split(',')]:
    t_img, t_ten = validation_transform(size)
    od_m = monai.metrics.DiceMetric(include_background=True)
    oc_m = monai.metrics.DiceMetric(include_background=True)
    od_l, oc_l = [], []
    with torch.no_grad():
        for fi, fm in zip(imgs, msks):
            img = np.asarray(Image.open(f'{vi}/{fi}'))
            m = np.asarray(Image.open(f'{vm}/{fm}'))
            m = (m[:, :, 0] if m.ndim == 3 else m).astype(np.int64)
            m[m == 0] = 2; m[m == 128] = 1; m[m == 255] = 0
            x = t_ten(image=t_img(image=img)['image'])['image'].unsqueeze(0).to(dev)
            y_true = torch.from_numpy(m).unsqueeze(0).to(dev)
            y_true = F.one_hot(y_true, num_classes=3).permute(0, 3, 1, 2)
            y_pred = model(x)
            y_pred = F.interpolate(y_pred, size=(y_true.size(2), y_true.size(3)), mode='bicubic')
            y_pred = torch.argmax(torch.softmax(y_pred, 1), dim=1)
            y_pred = F.one_hot(y_pred, num_classes=3).permute(0, 3, 1, 2)
            y_true[:, 1] = y_true[:, 1] + y_true[:, 2]
            y_pred[:, 1] = y_pred[:, 1] + y_pred[:, 2]
            od_m(y_pred[:, 1].unsqueeze(1), y_true[:, 1].unsqueeze(1))
            oc_m(y_pred[:, 2].unsqueeze(1), y_true[:, 2].unsqueeze(1))
            od_l.append(od_m.aggregate().item())
            oc_l.append(oc_m.aggregate().item())
    od, oc = np.mean(od_l), np.mean(oc_l)
    mean = np.mean([od, oc])
    d = abs(mean - rec['mean'])
    print(f'image_size {size}：OD {od:.10f}  OC {oc:.10f}  mean {mean:.10f}  '
          f'與 log 差 {d:.2e}  {"← 吻合" if d < 1e-4 else ""}')
