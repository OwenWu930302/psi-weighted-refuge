#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驗證 train_psi.py 的 LightDatasetFromRoot 與官方 DatasetFromRoot 輸出完全相同。

只取前 K 張（預設 40，train / val 各一組）建立暫存資料夾，兩種 Dataset 都只載入這 K 張，
因此不會 OOM。三層比對：
  1. 儲存的陣列：影像相同；原版 mask[:,:,0] 與省記憶體版 mask 數值相同
  2. 固定參數的幾何轉換（旋轉、翻轉、縮放，p=1）：兩種 mask 經轉換後相同
  3. 實際的 designed_transform / validation_transform，每張以相同亂數種子取出：
     送進模型的 (x, y) 逐元素相同

用法（FunduSegmenter 資料夾內）：
  conda activate fundusegmenter
  python3 check_light_dataset.py
"""
import os, sys, tempfile, argparse
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.datasets import DatasetFromRoot
from utils.transform import designed_transform, validation_transform
from train_psi import LightDatasetFromRoot

ap = argparse.ArgumentParser()
ap.add_argument('--data_root', default='/mnt/data2t/datasets/REFUGE/REFUGE')
ap.add_argument('--k', type=int, default=40)
ap.add_argument('--image_size', type=int, default=256)
a = ap.parse_args()

ALL_OK = True


def report(name, ok, extra=''):
    global ALL_OK
    ALL_OK &= ok
    print(f'  {"✓" if ok else "✗"} {name} {extra}')


def subset(split, tmp):
    src = f'{a.data_root}/{split}/ROIs'
    di, dm = os.path.join(tmp, split, 'images'), os.path.join(tmp, split, 'masks')
    os.makedirs(di); os.makedirs(dm)
    imgs = sorted(os.listdir(f'{src}/images'))[:a.k]
    msks = sorted(os.listdir(f'{src}/masks'))[:a.k]
    for f in imgs:
        os.symlink(f'{src}/images/{f}', os.path.join(di, f))
    for f in msks:
        os.symlink(f'{src}/masks/{f}', os.path.join(dm, f))
    return di, dm


def same(t1, t2):
    return t1.shape == t2.shape and torch.equal(torch.as_tensor(t1), torch.as_tensor(t2))


with tempfile.TemporaryDirectory() as tmp:
    for split, tf in [('training', designed_transform(a.image_size)),
                      ('validation', validation_transform(a.image_size))]:
        print(f'【{split}：前 {a.k} 張】')
        di, dm = subset(split, tmp)
        orig = DatasetFromRoot(di, dm, tf, 3)
        light = LightDatasetFromRoot(di, dm, tf, 3)

        # 1. 儲存的陣列
        ok_img = all(np.array_equal(x, y) for x, y in zip(orig.image_all, light.image_all))
        ok_msk = all(np.array_equal((m[:, :, 0] if m.ndim == 3 else m), l)
                     for m, l in zip(orig.mask_all, light.mask_all))
        mb_o = sum(m.nbytes for m in orig.mask_all) / 2**20
        mb_l = sum(m.nbytes for m in light.mask_all) / 2**20
        report('影像陣列相同', ok_img)
        report('mask 數值相同', ok_msk,
               f'（原版 {orig.mask_all[0].shape} {orig.mask_all[0].dtype} {mb_o:.0f} MB → '
               f'本版 {light.mask_all[0].shape} {light.mask_all[0].dtype} {mb_l:.0f} MB）')

        # 2. 固定參數的幾何轉換
        if split == 'training':
            ok = True
            for i, ang in enumerate(np.linspace(-15, 15, len(orig))):
                t = A.Compose([A.Rotate(limit=(ang, ang), interpolation=cv2.INTER_CUBIC, p=1),
                               A.VerticalFlip(p=1), A.HorizontalFlip(p=1),
                               A.Resize(height=a.image_size, width=a.image_size,
                                        interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_NEAREST),
                               ToTensorV2()])
                ro = t(image=orig.image_all[i], mask=orig.mask_all[i])
                rl = t(image=light.image_all[i], mask=light.mask_all[i])
                mo = ro['mask'][:, :, 0] if ro['mask'].ndim == 3 else ro['mask']
                ok &= same(ro['image'], rl['image']) and same(mo.long(), rl['mask'].long())
            report('固定旋轉 / 翻轉 / 縮放後相同', ok)

        # 3. 實際 transform，以相同種子逐張取出
        ti, tb = tf
        can_seed = hasattr(ti, 'set_random_seed') and hasattr(tb, 'set_random_seed')
        if not can_seed:
            print('  - 此版 albumentations 無 set_random_seed，第 3 層略過（第 1、2 層已涵蓋）')
            continue
        ok = True
        for i in range(len(orig)):
            ti.set_random_seed(1000 + i); tb.set_random_seed(2000 + i)
            xo, yo = orig[i]
            ti.set_random_seed(1000 + i); tb.set_random_seed(2000 + i)
            xl, yl = light[i]
            ok &= same(xo, xl) and same(yo, yl) and yo.dtype == yl.dtype
        report('實際 transform 輸出的 (x, y) 相同', ok, f'（y: {tuple(yl.shape)} {yl.dtype}）')

print('\n→ 全部 ✓：LightDatasetFromRoot 與官方 Dataset 輸出相同，可以開始訓練' if ALL_OK
      else '\n→ 有 ✗：不要開始訓練，把輸出貼回來')
