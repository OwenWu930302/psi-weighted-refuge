#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFUGE：FunduSegmenter 推論 → 每張 Dice + Ψ1 / Ψ5 / Ψ8（病例層級）

放在 FunduSegmenter 資料夾內執行（需要 model/ 與 utils/）。

與既有流程的對應（逐項沿用，不改算法）
------------------------------------------------------------
推論與還原   = test.py：ROI 影像 → validation_transform(224) → model
               → bicubic 插值到 800×800 → argmax → 依 centres.txt 貼回原圖尺寸
Dice         = test.py：在原圖尺寸上計算；OD = 類別 1+2，OC = 類別 2
               病例分數 dice = (OD + OC) / 2（recompute_caselevel.py）
預測圖       = test.py 的編碼：255 背景、128 視盤環、0 視杯
Ψ1 / Ψ5      = psi/run_all_psi.py 的 psi1_convex / psi5_exclusion 與 load_refuge()
               的遮罩定義（cup = 0，disc = 0|128，rim = 128），逐字複製
Ψ2           = run_psi2_psi8_refuge.py v3：視杯質心處垂直亮度衰減帶（與 CAMUS 相同參數），只比較視杯
Ψ8           = run_psi2_psi8_refuge.py v3：輸入旋轉 15°、AMP 推論、插值 800×800、
               轉回（最近鄰），只比較視杯
病例層級     = recompute_caselevel.py：各結構取 nanmax

逐張讀取，不預先載入整個資料集（避免 ReconstructTestDataset 的 OOM）。

用法
------------------------------------------------------------
  conda activate fundusegmenter
  cd ~/桌面/REFUGE/FunduSegmenter
  # 1) 先在 testing 上跑，與既有結果比對（驗證本程式與原流程一致）
  python3 refuge_infer_psi.py --split testing  --out ../psi_expert/check_testing
  # 2) 訓練集（產生加權清單、也是「加權前」的訓練集 Dice）
  python3 refuge_infer_psi.py --split training --out ../psi_expert/baseline_training
  # 3) 專家訓練完後，只需 Dice（不需 Ψ8）
  python3 refuge_infer_psi.py --split training --ckpt <專家>/saved_weights/best_weights.pth \
         --out ../psi_expert/expert_psi1_training --no_psi8
"""
import os, sys, csv, argparse, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.modules.adapters import PreAdapter, PostAdapter
from model.modules.vit import VisionTransformer
from model.modules.segmenter_decoder import SegmenterDecoder
from model.fundusegmenter import FunduSegmenter
from utils.transform import validation_transform

IMAGE_SIZE = 224       # 預設同 test.py 與 Ψ8 v3；可用 --image_size 覆寫
ROI_SIZE   = 800        # test.py：非 IDRiD 一律插值到 800×800
OUT_CH     = 3
NORM       = 'bn'
ANGLE      = 15         # run_psi2_psi8_refuge.py v3
# Ψ2 的 V_shadow 參數（run_psi2_psi8_refuge.py v3，與 CAMUS 相同）
SHADOW_A   = 3.0
BAND_FRAC  = 0.25
TAPER_FRAC = 0.15
CUP        = 2


# ------------------------------------------------------------------
# Ψ：逐字複製自 psi-weighted-nnunet/psi/run_all_psi.py
# ------------------------------------------------------------------
def psi1_convex(mask):
    """Ψ1 凸包比 = 凸包面積 / 實際面積。3D 逐切片以面積加權平均。"""
    if mask.sum() == 0:
        return np.nan
    if mask.ndim == 2:
        slices = [mask]
    else:
        slices = [mask[:, :, z] for z in range(mask.shape[2])]
    ta = th = 0.0
    for s in slices:
        if s.sum() < 10:
            continue
        cs, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        b = max(cs, key=cv2.contourArea)
        a = cv2.contourArea(b)
        if a < 10:
            continue
        ta += a
        th += cv2.contourArea(cv2.convexHull(b))
    return float(th / ta) if ta > 0 else np.nan


def psi5_exclusion(target, others):
    """Ψ5 互斥 = 該結構與其他所有結構的最大重疊比。"""
    if target is None or target.sum() == 0:
        return np.nan
    m = 0.0
    for o in others:
        if o is None:
            continue
        u = (target | o).sum()
        if u > 0:
            m = max(m, (target & o).sum() / u)
    return float(m)


def nanmax(*v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return max(v) if v else np.nan


def dice_bin(a, b):
    """二元 Dice。GT 為空時回傳 NaN（同 monai DiceMetric 預設 ignore_empty）。"""
    if b.sum() == 0:
        return np.nan
    return float(2.0 * (a & b).sum() / (a.sum() + b.sum()))


def build_shadow(h, w, i0, cx):
    """自 i0 列向下、於 cx 為中心的垂直帶內指數衰減（逐字複製自 v3）。"""
    att = np.ones((h, w), dtype=np.float32)
    half = max(1, int(w * BAND_FRAC / 2))
    j0, j1 = max(0, cx - half), min(w, cx + half)
    bw = j1 - j0
    if bw <= 0:
        return att
    rows = np.arange(h, dtype=np.float32)
    denom = max(1.0, float(h - i0))
    vert = np.where(rows >= i0,
                    np.exp(-SHADOW_A * (rows - i0) / denom), 1.0).astype(np.float32)
    taper = max(1, int(bw * TAPER_FRAC))
    prof = np.ones(bw, dtype=np.float32)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper, dtype=np.float32)))
    prof[:taper] = ramp
    prof[-taper:] = ramp[::-1]
    att[:, j0:j1] = 1.0 - np.outer(1.0 - vert, prof)
    return att


def dice_v3(a, b):
    """run_psi2_psi8_refuge.py v3 的 dice（兩者皆空 → 1.0）。"""
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


# ------------------------------------------------------------------
def build_model():
    pre  = PreAdapter(input_channel=3, mid_channel=64, norm=NORM)
    enc  = VisionTransformer(image_size=(224, 224), patch_size=16, n_layers=24,
                             d_model=1024, d_ff=4096, n_heads=16, n_cls=1000,
                             dropout=0.0, drop_path_rate=0.0, distilled=False, channels=3)
    dec  = SegmenterDecoder(n_cls=OUT_CH, patch_size=16, d_encoder=1024, n_layers=2,
                            n_heads=16, d_model=1024, d_ff=4096,
                            drop_path_rate=0.0, dropout=0.1)
    post = PostAdapter(input_channel=OUT_CH, mid_channel=64, output_channel=OUT_CH, norm=NORM)
    return FunduSegmenter(pre_adapter=pre, encoder=enc, decoder=dec, post_adapter=post)


def read_centres(path):
    """centres.txt：第一行為標題，其後每行 name,x,y,h,w（同 ReconstructTestDataset）。"""
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            p = line.strip().split(',')
            if len(p) < 5:
                continue
            d[p[0][:-4]] = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
    return d


def restore(pred_roi, x, y, h_ori, w_ori):
    """與 test.py 完全相同的貼回邏輯。"""
    out = np.zeros((h_ori, w_ori), dtype=np.uint8)
    left, right, top, bottom = x - 400, x + 400, y - 400, y + 400
    rt, rb = max(0, top), min(h_ori, bottom)
    rl, rr = max(0, left), min(w_ori, right)
    ct, cl = max(0, -top), max(0, -left)
    out[rt:rb, rl:rr] = pred_roi[ct:ct + (rb - rt), cl:cl + (rr - rl)]
    return out


def find_mask(mask_dir, stem):
    for ext in ('.bmp', '.png', '.jpg', '.tif'):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/mnt/data2t/datasets/REFUGE/REFUGE')
    ap.add_argument('--split', required=True, choices=['training', 'validation', 'testing'])
    ap.add_argument('--ckpt', default='repro_notes/best_designed.pth')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_psi8', action='store_true', help='不算 Ψ2 / Ψ8（需重新推論的兩項），只算 Dice 與 Ψ1/Ψ5')
    ap.add_argument('--no_maps', action='store_true', help='不存預測圖')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--image_size', type=int, default=IMAGE_SIZE,
                    help='模型輸入尺寸；必須與 baseline 測試時相同（以 check 步驟確認）')
    ap.add_argument('--psi8_size', type=int, default=0,
                    help='Ψ2 / Ψ8 推論用的輸入尺寸；0 = 與 --image_size 相同。原 v3 為 224')
    args = ap.parse_args()

    S = os.path.join(args.data_root, args.split)
    roi_dir, mask_dir = f'{S}/ROIs/images', f'{S}/masks'
    centres = read_centres(f'{S}/ROIs/centres.txt')
    map_dir = os.path.join(args.out, 'map')
    os.makedirs(map_dir, exist_ok=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    print(f'權重 {args.ckpt}（epoch {ck.get("epoch")}）  裝置 {dev}')
    del ck
    model.to(dev).eval()

    t_img, t_ten = validation_transform(args.image_size)
    psi8_size = args.psi8_size or args.image_size
    t8_img, t8_ten = validation_transform(psi8_size)
    files = sorted(f for f in os.listdir(roi_dir) if f.lower().endswith(('.png', '.jpg')))
    if args.limit:
        files = files[:args.limit]
    miss = [f for f in files if f[:-4] not in centres]
    if miss:
        sys.exit(f'centres.txt 缺少 {len(miss)} 張，例如 {miss[:3]}')
    print(f'{args.split}：{len(files)} 張　輸入尺寸 {args.image_size}'
          + ('' if args.no_psi8 else f'　Ψ2 / Ψ8 輸入尺寸 {psi8_size}'))

    def infer(x, amp):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        out = F.interpolate(out.float(), size=(ROI_SIZE, ROI_SIZE), mode='bicubic')
        return torch.argmax(torch.softmax(out, 1), dim=1)[0].to(torch.uint8).cpu().numpy()

    rows, t0 = [], time.time()
    with torch.no_grad():
        for k, fn in enumerate(files):
            stem = fn[:-4]
            img = np.asarray(Image.open(os.path.join(roi_dir, fn)))
            x = t_ten(image=t_img(image=img)['image'])['image'].unsqueeze(0).to(dev)

            # ---- test.py：fp32 推論 → 800×800 → 貼回原圖 ----
            p_roi = infer(x, amp=False)
            cx, cy, h_ori, w_ori = centres[stem]
            p = restore(p_roi, cx, cy, h_ori, w_ori)

            # ---- Dice（原圖尺寸）----
            mp = find_mask(mask_dir, stem)
            if mp is None:
                sys.exit(f'找不到 {stem} 的原圖標註（{mask_dir}）')
            g = np.asarray(Image.open(mp))
            g = g[..., 0] if g.ndim == 3 else g
            if g.shape != p.shape:
                sys.exit(f'{stem}：標註 {g.shape} 與還原後預測 {p.shape} 尺寸不同')
            od_dice = dice_bin(p >= 1, (g == 0) | (g == 128))
            oc_dice = dice_bin(p == 2, g == 0)

            # ---- 預測圖（test.py 編碼）與 Ψ1 / Ψ5（load_refuge 的遮罩定義）----
            a = np.full(p.shape, 255, np.uint8)
            a[p == 1] = 128
            a[p == 2] = 0
            if not args.no_maps:
                cv2.imwrite(os.path.join(map_dir, stem + '.png'), a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cup, disc, rim = (a == 0), (a == 0) | (a == 128), (a == 128)
            p1_oc, p1_od = psi1_convex(cup), psi1_convex(disc)
            p5_oc, p5_od = psi5_exclusion(cup, [rim]), psi5_exclusion(disc, [cup])

            r = dict(case=stem, od_dice=od_dice, oc_dice=oc_dice,
                     dice=np.nanmean([od_dice, oc_dice]),
                     psi1_convex=nanmax(p1_oc, p1_od), psi5_exclusion=nanmax(p5_oc, p5_od),
                     psi1_OC=p1_oc, psi1_OD=p1_od, psi5_OC=p5_oc, psi5_OD=p5_od,
                     cup_px=int(cup.sum()), disc_px=int(disc.sum()))

            # ---- Ψ8：run_psi2_psi8_refuge.py v3 ----
            if not args.no_psi8:
                amp = dev.type == 'cuda'
                x8 = x if psi8_size == args.image_size else \
                    t8_ten(image=t8_img(image=img)['image'])['image'].unsqueeze(0).to(dev)
                cup0 = infer(x8, amp) == CUP
                pr = infer(TF.rotate(x8, ANGLE, interpolation=TF.InterpolationMode.BILINEAR), amp)
                pr_t = torch.from_numpy(pr).unsqueeze(0).unsqueeze(0).float()
                back = TF.rotate(pr_t, -ANGLE, interpolation=TF.InterpolationMode.NEAREST)
                back = back.squeeze().numpy().astype(np.uint8)
                r['psi8_rotation'] = round(1.0 - dice_v3(cup0, back == CUP), 6)
                # ---- Ψ2：亮度衰減 → 推論（v3 逐行對應）----
                h, w = x8.shape[-2], x8.shape[-1]
                fg = np.argwhere(cup0)
                if len(fg) == 0:
                    i0, cx = h // 2, w // 2
                else:
                    sy, sx = h / cup0.shape[0], w / cup0.shape[1]   # 800×800 → 輸入尺寸
                    i0 = int(fg[:, 0].mean() * sy)
                    cx = int(fg[:, 1].mean() * sx)
                att = torch.from_numpy(build_shadow(h, w, i0, cx)).to(dev).view(1, 1, h, w)
                ps = infer(x8 * att, amp)
                r['psi2_shadow'] = round(1.0 - dice_v3(cup0, ps == CUP), 6)
                del x8, att, ps
            rows.append(r)
            del img, x, p, g, a

            if (k + 1) % 50 == 0 or k + 1 == len(files):
                el = time.time() - t0
                print(f'  {k+1}/{len(files)}  {el/(k+1):.2f} s/張  '
                      f'目前 mean Dice {np.nanmean([q["dice"] for q in rows]):.4f}', flush=True)

    out_csv = os.path.join(args.out, 'per_image.csv')
    keys = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    od = np.nanmean([q['od_dice'] for q in rows]); oc = np.nanmean([q['oc_dice'] for q in rows])
    print(f'完成：{out_csv}\n  OD {od:.4f}  OC {oc:.4f}  mean {np.nanmean([od, oc]):.4f}')


if __name__ == '__main__':
    main()
