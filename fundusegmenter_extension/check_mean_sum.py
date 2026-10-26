"""
驗證：拿掉 / w.sum() 之後，.mean() 和 .sum() 哪一個等於官方 train.py 的 loss

官方 train.py 第 389 行：loss = dice_loss(y_pred, y_true) + ce_loss(y_pred, y_true)
改法 B：loss = (w * (l_dice + l_ce)).mean()
改法 A：loss = (w * (l_dice + l_ce)).sum()

官方沒有加權，所以比較時 w 全為 1（= 對照組的情況）。
每個 batch 只做一次 forward，三種 loss 用同一個 y_pred，分別求梯度，比較：
  - loss 的數值
  - 梯度（所有可訓練參數）：比值與最大差

與訓練相同的部分：
  - 模型：FunduSegmenter（凍結 encoder、adapter 可訓練，同 train.py 第 231-237 行），載入 baseline 權重，train 模式
  - 資料：training ROI 400 張、designed_transform(256)、batch 16、shuffle、drop_last（同 train.py）
  - loss 定義：逐字複製自 train_psi.py 第 439-443 行；計算：第 506-507 行

用法（放在 FunduSegmenter 資料夾，conda activate fundusegmenter）：
  python3 check_mean_sum.py                       # 跑完一個 epoch（25 個 batch）
  python3 check_mean_sum.py --batches 5           # 只跑 5 個 batch
"""
import argparse

import monai
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from refuge_infer_psi import build_model
from train_psi import LightDatasetFromRoot
from utils.set_seed import set_seed
from utils.transform import designed_transform

p = argparse.ArgumentParser()
p.add_argument('--data_root', default='/mnt/data2t/datasets/REFUGE/REFUGE')
p.add_argument('--ckpt', default='repro_notes/best_designed.pth')
p.add_argument('--batch_size', default=16, type=int)
p.add_argument('--batches', default=0, type=int, help='0 = 一整個 epoch')
a = p.parse_args()
set_seed(112316)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 模型（與 train.py 相同的凍結方式） ----------
model = build_model()
model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=False)['model'], strict=True)
for q in model.encoder.parameters():
    q.requires_grad = False
for block in model.encoder.blocks:
    for q in block.adapter.parameters():
        q.requires_grad = True
need_grad = [q for q in model.parameters() if q.requires_grad]
model = model.to(dev).train()

# ---------- 資料（與 train.py 相同） ----------
ds = LightDatasetFromRoot(f'{a.data_root}/training/ROIs/images', f'{a.data_root}/training/ROIs/masks',
                          designed_transform(256), 3)
loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=1, drop_last=True)

# ---------- loss 定義（逐字複製自 train_psi.py） ----------
dice_loss = monai.losses.DiceLoss(include_background=True, squared_pred=False, softmax=True)
ce_loss = torch.nn.CrossEntropyLoss()
dice_loss_ps = monai.losses.DiceLoss(include_background=True, squared_pred=False, softmax=True, reduction='none')
ce_loss_ps = torch.nn.CrossEntropyLoss(reduction='none')

n = a.batches or len(loader)
print(f'裝置 {dev}　權重 {a.ckpt}　{n} 個 batch × {a.batch_size} 張　可訓練參數 {sum(q.numel() for q in need_grad) / 1e6:.1f} M\n')
print(f'{"batch":<6}{"官方 loss":>13}{".mean()":>13}{".sum()":>13}'
      f'{"mean÷官方":>11}{"sum÷官方":>10}{"梯度 mean÷官方":>16}{"梯度 sum÷官方":>15}')

rows = []
for b, (x, y_true) in enumerate(loader):
    if b >= n:
        break
    x, y_true = x.to(dev), y_true.to(dev)
    y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=3).permute(0, 3, 1, 2)
    y_pred = model(x)                                       # 只 forward 一次，三種 loss 共用
    w = torch.ones(len(x), device=dev)                      # 官方沒有加權 → w 全為 1

    loss_off = dice_loss(y_pred, y_true.float()) + ce_loss(y_pred, y_true.float())   # 官方 train.py 第 389 行
    l_dice = dice_loss_ps(y_pred, y_true.float()).flatten(1).mean(1)
    l_ce = ce_loss_ps(y_pred, y_true.float()).flatten(1).mean(1)
    loss_mean = (w * (l_dice + l_ce)).mean()                # 改法 B
    loss_sum = (w * (l_dice + l_ce)).sum()                  # 改法 A

    def grads(loss):
        return torch.cat([g.flatten() for g in torch.autograd.grad(loss, need_grad, retain_graph=True)])
    g_off, g_mean, g_sum = grads(loss_off), grads(loss_mean), grads(loss_sum)

    r = dict(off=loss_off.item(), mean=loss_mean.item(), sum=loss_sum.item(),
             g_mean=(g_mean.norm() / g_off.norm()).item(), g_sum=(g_sum.norm() / g_off.norm()).item(),
             dg_mean=(g_mean - g_off).abs().max().item(), dg_sum=(g_sum - g_off).abs().max().item(),
             dg_sum16=(g_sum / len(x) - g_off).abs().max().item(), gmax=g_off.abs().max().item())
    rows.append(r)
    print(f'{b + 1:<6}{r["off"]:>13.8f}{r["mean"]:>13.8f}{r["sum"]:>13.8f}'
          f'{r["mean"] / r["off"]:>11.6f}{r["sum"] / r["off"]:>10.4f}{r["g_mean"]:>16.6f}{r["g_sum"]:>15.4f}')
    del x, y_true, y_pred, l_dice, l_ce, loss_off, loss_mean, loss_sum, g_off, g_mean, g_sum   # 釋放計算圖
    if dev == 'cuda':
        torch.cuda.empty_cache()

mx = lambda k: max(r[k] for r in rows)
B = a.batch_size
print(f'\n========== 結論（{len(rows)} 個 batch） ==========')
print(f'{"":<10}{"loss 與官方的最大差":>22}{"梯度與官方的最大差":>22}{"等於官方":>10}')
for k in ('mean', 'sum'):
    dl = max(abs(r[k] - r['off']) for r in rows)
    dg = mx(f'dg_{k}')
    ok = dl < 1e-5 and dg < 1e-4 * mx('gmax')
    print(f'{"." + k + "()":<10}{dl:>22.2e}{dg:>22.2e}{"✓" if ok else "✗":>10}')
print(f'\n.sum() ÷ 官方：loss {min(r["sum"] / r["off"] for r in rows):.4f} ~ {max(r["sum"] / r["off"] for r in rows):.4f}'
      f'，梯度 {min(r["g_sum"] for r in rows):.4f} ~ {mx("g_sum"):.4f}（= batch 張數 {B}）')
print(f'.sum() ÷ {B} 之後與官方梯度的最大差：{mx("dg_sum16"):.2e}（官方梯度最大值 {mx("gmax"):.2e}）')
print('（差在 1e-7 量級為 float32 捨入誤差）')
