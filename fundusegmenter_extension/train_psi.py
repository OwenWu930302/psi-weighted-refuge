"""
[Ψ加權] 由 FunduSegmenter 官方 train.py 複製修改，所有改動以 [Ψ加權] 標示。
另有 [記憶體] 標示的省記憶體載入（輸出不變）。用 diff train.py train_psi.py 可檢視全部改動。

MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import argparse
import os
import monai
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.fundusegmenter import FunduSegmenter
from model.modules.vit import VisionTransformer
from model.modules.segmenter_decoder import SegmenterDecoder
from model.modules.adapters import PreAdapter, PostAdapter
from model.utils.pos_embed import interpolate_pos_embed
from model.baselines.retfoundsegmenter import RETFoundSegmenter
from model.baselines.dunet import DUNet
from model.baselines.transunet.networks.vit_seg_modeling import VisionTransformer as TransUNet
from model.baselines.transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from utils.set_seed import set_seed
from utils.datasets import DatasetFromRoot, DatasetFromList
from utils.transform import no_transform, spatial_transform, designed_transform, validation_transform
from utils.random_split import random_split
from PIL import Image                                       # [記憶體]


def get_args_parser():
    parser = argparse.ArgumentParser('FunduSegmenter for OD/OC segmentation', add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lr', default=0.001, type=float, 
                       help='learning rate')
    
    # Model parameters
    parser.add_argument('--model_selection', default='FunduSegmenter', type=str,
                        help='select model from FunduSegmenter, baseline_RETFoundSegmenter, baseline_DUNet or baseline_TransUNet')
    parser.add_argument('--encoder_path', default='./RETFound_mae_natureCFP.pth', type=str,
                        help='RETFound weights path')
    parser.add_argument('--transunet_pretrained_path', default='./imagenet21k_R50+ViT-B_16.npz', type=str,
                        help='TransUNet pretrained weight path')
    parser.add_argument('--output_channel', default=3, type=int, 
                        help='the number of model output channels')
    parser.add_argument('--norm', default='bn', type=str,
                        help='select normalization for adapters, bn or gn')
    
    # Dataset parameters
    parser.add_argument('--label_n_cls', default=3, type=int, 
                        help='the number of label classes')
    parser.add_argument('--seed', default=112316, type=int, 
                        help='set seed for reproducibility')
    parser.add_argument('--train_image_path', default='./datasets/Vampire/training/ROIs/images', type=str,
                        help='training image folder path')
    parser.add_argument('--train_mask_path', default='./datasets/Vampire/training/ROIs/masks', type=str,
                        help='training mask folder path')
    parser.add_argument('--separate_val', default=0, type=int,
                       help='choose if validation data is separate, 1 for True, 0 for False')
    parser.add_argument('--val_image_path', default='./datasets/Vampire/validation/ROIs/images', type=str,
                        help='validation image folder path')
    parser.add_argument('--val_mask_path', default='./datasets/Vampire/validation/ROIs/masks', type=str,
                        help='validation mask folder path')
    parser.add_argument('--num_workers', default=1, type=int)    
    parser.add_argument('--image_size', default=256, type=int, 
                        help='input image size')
    parser.add_argument('--transform_mode', default='no_transform', type=str,
                        help='select data transform: no_transform, basic_transform or designed_transform')
    parser.add_argument('--train_split_rate', default=0.8, type=float, 
                       help='training data split rate if no separate validation data')
    
    # Output parameters
    parser.add_argument('--results_base_dir', default='./results', type=str,
                        help='directory for saving results')
    parser.add_argument('--weights_dir', default='./saved_weights', type=str,
                        help='directory for saving weights')
    parser.add_argument('--tb_path', default='./tensorboard', type=str,
                        help='path for saving tensorboard')
    parser.add_argument('--records_figure_path', default='./Records.svg', type=str,
                        help='path for saving loss and dice figure')
    # [Ψ加權] 加權清單（csv，需有 case 欄，值為不含副檔名的影像檔名）與倍率
    parser.add_argument('--psi_cases', default='', type=str,
                        help='[Ψ加權] 要加權的病例清單 csv；留空 = 不加權（與 train.py 相同）')
    parser.add_argument('--psi_factor', default=3.0, type=float,
                        help='[Ψ加權] 清單病例的 loss 倍率')
    parser.add_argument('--psi_norm', default='wsum', choices=['wsum', 'mean'],                    # 新增
                        help='[Ψ加權] batch 內的平均方式：wsum = Σ w L / Σ w；mean = Σ w L / B（w 全為 1 時兩者皆等於官方 loss）')
    parser.add_argument('--grad_ckpt', default=0, type=int,
                        help='[記憶體] 1 = RETFound encoder 逐 block gradient checkpointing（結果不變，較慢、省 GPU 記憶體）')
    return parser


class LightDatasetFromRoot(DatasetFromRoot):
    """[記憶體] 與 utils.datasets.DatasetFromRoot 相同，只差 mask 的儲存方式。

    原版：mask 以 np.asarray(...).astype(int) 保存 → ROI mask 為 800×800×3 int64，每張 15 MB，
          800 張約 12 GB，15 GB 記憶體的機器會被 OOM killer 砍掉。
    本版：讀入時即取第 0 通道並以 uint8 保存（每張 0.6 MB）。
    等價性：原版 __getitem__ 最後只取 mask[:,:,0]；幾何轉換對 mask 為最近鄰、逐通道獨立；
           類別值 0/1/2 在 uint8 與 int64 下相同；最後皆 .long()。
           → 送進模型的 (x, y) 完全相同，由 check_light_dataset.py 逐張驗證。
    """
    def __init__(self, image_root_path, mask_root_path, transform, n_cls):
        self.image_all = []
        self.mask_all = []
        image_list = os.listdir(image_root_path)
        image_list.sort()
        for i in image_list:
            self.image_all.append(np.asarray(Image.open(image_root_path + os.sep + i)))
        mask_list = os.listdir(mask_root_path)
        mask_list.sort()
        for j in mask_list:
            m = np.asarray(Image.open(mask_root_path + os.sep + j))
            if m.ndim == 3:
                m = m[:, :, 0]
            m = m.astype(np.uint8)        # 原版為 .astype(int)
            if n_cls == 2:
                m[m == 255] = 1
            elif n_cls == 3:
                m[m == 0] = 2
                m[m == 128] = 1
                m[m == 255] = 0
            else:
                raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            self.mask_all.append(m)
        self.transform_image, self.transform_both = transform


def enable_grad_ckpt(model):
    """[記憶體] 對 RETFound encoder 的每個 block 做 gradient checkpointing。

    反向傳播時重新計算 block 的前向，而不是保存所有中間值。
    encoder 以 dropout=0.0、drop_path_rate=0.0 建立，block 內沒有任何隨機性，
    重新計算的值與第一次相同 → loss 與梯度不變（由 check_grad_ckpt.py 驗證）。
    以覆寫實例的 forward 實作，模組結構與 state_dict 的鍵名完全不變，
    存下的權重可直接給 test.py / refuge_infer_psi.py 使用。
    """
    from torch.utils.checkpoint import checkpoint
    blocks = getattr(getattr(model, 'encoder', None), 'blocks', None)
    if blocks is None:
        raise ValueError('[記憶體] 此模型沒有 encoder.blocks，無法使用 --grad_ckpt')
    for blk in blocks:
        f = blk.forward
        def wrapped(x, _f=f):
            if torch.is_grad_enabled():
                return checkpoint(_f, x, use_reentrant=False)
            return _f(x)
        blk.forward = wrapped
    return len(blocks)


class PsiWeightedDataset(torch.utils.data.Dataset):
    """[Ψ加權] 不改變原資料集的內容與順序，只多回傳一個權重。"""
    def __init__(self, base, names, cases, factor):
        self.base = base
        self.w = np.array([factor if n in cases else 1.0 for n in names], dtype=np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, torch.tensor(self.w[idx])


def main(args):
    set_seed(args.seed)
    image_size = args.image_size
    output_channel = args.output_channel
    norm = args.norm

    ######################################## Model architecture ########################################
    # FunduSegmenter
    if args.model_selection == 'FunduSegmenter':
        # pre-adapter setting
        pre_adapter = PreAdapter(input_channel=3, mid_channel=64, norm=norm)        
        # load RETFound
        encoder = VisionTransformer(image_size=(224,224),
                                    patch_size=16,
                                    n_layers=24,
                                    d_model=1024,
                                    d_ff=4096,
                                    n_heads=16,
                                    n_cls=1000,
                                    dropout=0.0,
                                    drop_path_rate=0.0,
                                    distilled=False,
                                    channels=3,
                                   )       
        # load RETFound weights
        checkpoint = torch.load(args.encoder_path, map_location='cpu', weights_only=False)
        checkpoint_encoder = checkpoint['model']
        state_dict = encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_encoder and checkpoint_encoder[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_encoder[k]
        # interpolate position embedding
        interpolate_pos_embed(encoder, checkpoint_encoder)
        # load pre-trained model
        msg = encoder.load_state_dict(checkpoint_encoder, strict=False)    
        # create decoder
        decoder = SegmenterDecoder(n_cls=output_channel,
                                   patch_size=16,
                                   d_encoder=1024,
                                   n_layers=2,
                                   n_heads=16,
                                   d_model=1024,
                                   d_ff=4096,
                                   drop_path_rate=0.0,
                                   dropout=0.1,
                                   )    
        # post-adapter setting
        post_adapter = PostAdapter(input_channel=output_channel, mid_channel=64, output_channel=output_channel, norm=norm)    
        # create model
        model = FunduSegmenter(pre_adapter=pre_adapter, encoder=encoder, decoder=decoder, post_adapter=post_adapter)       
        # freeze the encoder
        for p in model.encoder.parameters():
            p.requires_grad = False 
        for block in model.encoder.blocks:
            for p in block.adapter.parameters():
                p.requires_grad = True
        need_grad = [p for p in model.parameters() if p.requires_grad] 
        # calculate parameter amounts and print
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        retfound_parameters = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        preadapter_parameters = sum(p.numel() for p in model.pre_adapter.parameters()) / 1e6
        decoder_parameters = sum(p.numel() for p in model.decoder.parameters()) / 1e6
        postadapter_parameters = sum(p.numel() for p in model.post_adapter.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of frozen parameters of RETFound (M): %.3f' % (retfound_parameters))
        print('Number of pre_adapter parameters (M): %.3f' % (preadapter_parameters))
        print('Number of decoder parameters (M): %.3f' % (decoder_parameters))
        print('Number of post_adapter parameters (M): %.3f' % (postadapter_parameters))    
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters))
    
    # ablation experiment baseline: RETFoundSegmenter
    elif args.model_selection == 'baseline_RETFoundSegmenter':       
        # load RETFound
        encoder = VisionTransformer(image_size=(224,224),
                                    patch_size=16,
                                    n_layers=24,
                                    d_model=1024,
                                    d_ff=4096,
                                    n_heads=16,
                                    n_cls=1000,
                                    dropout=0.0,
                                    drop_path_rate=0.0,
                                    distilled=False,
                                    channels=3,
                                   )      
        # load RETFound weights
        checkpoint = torch.load(args.encoder_path, map_location='cpu', weights_only=False)
        checkpoint_encoder = checkpoint['model']
        state_dict = encoder.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_encoder and checkpoint_encoder[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_encoder[k]
        # interpolate position embedding
        interpolate_pos_embed(encoder, checkpoint_encoder)
        # load pre-trained model
        msg = encoder.load_state_dict(checkpoint_encoder, strict=False)    
        # create decoder
        decoder = SegmenterDecoder(n_cls=output_channel,
                                   patch_size=16,
                                   d_encoder=1024,
                                   n_layers=2,
                                   n_heads=16,
                                   d_model=1024,
                                   d_ff=4096,
                                   drop_path_rate=0.0,
                                   dropout=0.1,
                                   )        
        # create model
        if image_size != 224:
            raise ValueError('Input size of baseline_RETFoundSegmenter must be 224*224')
        else:            
            model = RETFoundSegmenter(encoder=encoder, decoder=decoder)       
        # freeze the encoder
        for p in model.encoder.parameters():
            p.requires_grad = False    
        need_grad = [p for p in model.parameters() if p.requires_grad] 
        # calculate parameter amounts and print
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        retfound_parameters = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        decoder_parameters = sum(p.numel() for p in model.decoder.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of frozen parameters of RETFound (M): %.3f' % (retfound_parameters))
        print('Number of decoder parameters (M): %.3f' % (decoder_parameters))
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters))   

    # DUNet
    elif args.model_selection == 'baseline_DUNet':
        # create decoder
        model = DUNet(input_channel=3, output_channel=output_channel, base_channel=32)
        # calculate parameter amounts and print
        need_grad = [p for p in model.parameters() if p.requires_grad]         
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6        
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters)) 

    # TransUNet
    elif args.model_selection == 'baseline_TransUNet':
        # create decoder
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = output_channel
        config_vit.n_skip = 3 
        config_vit.pretrained_path = args.transunet_pretrained_path
        config_vit.patches.grid = (int(image_size / 16), int(image_size / 16))
        model = TransUNet(config_vit, img_size=image_size, num_classes=config_vit.n_classes).cuda()
        # load pre-trained model
        model.load_from(weights=np.load(config_vit.pretrained_path))
        # calculate parameter amounts and print
        need_grad = [p for p in model.parameters() if p.requires_grad]         
        total_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        learnable_parameters = sum(p.numel() for p in need_grad) / 1e6        
        print("Model = %s" % str(model))
        print('Total number of parameters (M): %.3f' % (total_parameters))
        print('Number of learnable parameters (M): %.3f' % (learnable_parameters)) 
    else:
        raise ValueError('Please select correct model')

    # [記憶體] encoder 逐 block gradient checkpointing（選用，見 enable_grad_ckpt）
    if args.grad_ckpt:
        n = enable_grad_ckpt(model)
        print(f'[記憶體] 已啟用 encoder gradient checkpointing：{n} 個 block')

    ######################################## Datasets ########################################
    # load datasets
    separate_val = args.separate_val
    label_n_cls = args.label_n_cls    
    
    # choose data transform strategy
    transform_mode = args.transform_mode
    val_transform = validation_transform(image_size)
    if transform_mode == 'no_transform':
        train_transform = no_transform(image_size)
    elif transform_mode == 'spatial_transform':
        train_transform = spatial_transform(image_size)
    elif transform_mode == 'designed_transform':
        train_transform = designed_transform(image_size)
    else: 
        raise ValueError('Please select correct transform mode from \'no_transform\', \'spatial_transform\', or \'designed_transform\'.')       
    
    # load data if the validation dataset are separate
    if separate_val == 1:        
        train_dataset = LightDatasetFromRoot(image_root_path = args.train_image_path,   # [記憶體] 
                                        mask_root_path = args.train_mask_path, 
                                        transform = train_transform,
                                        n_cls = label_n_cls,
                                       )
        val_dataset = LightDatasetFromRoot(image_root_path = args.val_image_path,   # [記憶體] 
                                      mask_root_path = args.val_mask_path, 
                                      transform = val_transform,
                                      n_cls = label_n_cls,
                                     )
    
    # load data if the validation dataset are not separate
    elif separate_val == 0:
        # split train and validation sets
        train_image_list, train_mask_list, val_image_list, val_mask_list = random_split(args.train_image_path, 
                                                                                        args.train_mask_path, 
                                                                                        args.train_split_rate,
                                                                                       )
        train_dataset = DatasetFromList(image_list = train_image_list, 
                                        mask_list = train_mask_list, 
                                        transform = train_transform,
                                        n_cls = label_n_cls,
                                       )
        val_dataset = DatasetFromList(image_list = val_image_list, 
                                      mask_list = val_mask_list, 
                                      transform = val_transform,
                                      n_cls = label_n_cls,
                                     )
    else:
        raise ValueError('Please input 1 for the separate validation dataset, or input 0 for the mixed datasets.')        
    print(f'Number of train images: {len(train_dataset)}')
    print(f'Number of validation images: {len(val_dataset)}')

    # [Ψ加權] 依與 Dataset 相同的排序取得每張的檔名，包成回傳 (x, y, w) 的資料集
    if separate_val == 1:
        train_names = sorted(os.listdir(args.train_image_path))
    else:
        train_names = [os.path.basename(p) for p in train_image_list]   # DatasetFromList 已就地排序
    train_names = [os.path.splitext(n)[0] for n in train_names]
    assert len(train_names) == len(train_dataset), '[Ψ加權] 檔名數與資料集張數不符'
    psi_cases = set()
    if args.psi_cases:
        import csv as _csv
        with open(args.psi_cases, newline='') as f:
            psi_cases = {os.path.splitext(r['case'].strip())[0] for r in _csv.DictReader(f)}
        missing = sorted(psi_cases - set(train_names))
        if missing:
            raise ValueError(f'[Ψ加權] 清單中 {len(missing)} 例不在訓練集：{missing[:5]}')
    train_dataset = PsiWeightedDataset(train_dataset, train_names, psi_cases, args.psi_factor)
    print(f'[Ψ加權] 清單 {args.psi_cases or "（無）"}　case 數 {len(psi_cases)}　'
          f'倍率 {args.psi_factor if psi_cases else 1.0}　平均方式 {args.psi_norm}　'
          f'訓練集中被加權 {int((train_dataset.w > 1).sum())} / {len(train_dataset)} 張')
        
    # dataloader
    batch_size = args.batch_size
    num_workers = args.num_workers    
    loader_train = DataLoader(train_dataset,                           
                              batch_size = batch_size,                           
                              shuffle = True,                           
                              num_workers = num_workers,                        
                              drop_last = True,                         
                             )
    loader_val = DataLoader(val_dataset,                           
                            batch_size = 1,                           
                            shuffle = False,                           
                            num_workers = num_workers,                        
                            drop_last = False,                         
                           )
    
    ######################################## Implementations ########################################
    # loss function
    dice_loss = monai.losses.DiceLoss(include_background=True, squared_pred=False, softmax=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    # [Ψ加權] 逐張版本：與上面兩個完全相同，只是不先平均
    dice_loss_ps = monai.losses.DiceLoss(include_background=True, squared_pred=False, softmax=True, reduction='none')
    ce_loss_ps = torch.nn.CrossEntropyLoss(reduction='none')
    psi_checked = False
    n_weighted_epoch = 0
    # epochs
    epochs = args.epochs
    # optimizer
    optimizer = optim.AdamW(need_grad, lr = args.lr, weight_decay=0.001)

    ######################################## Training ########################################   
    # create results directory
    os.makedirs(args.results_base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_base_dir, args.weights_dir), exist_ok=True)

    # start training
    print(f'If GPU is available: {torch.cuda.is_available()}')    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
    model.to(device)        
    
    loss_train_all = []
    od_dice_all = []
    oc_dice_all = []
    dice_mean_all = []
    loss_train = []
    od_dice = []
    oc_dice = []    
    lr_all = []
    max_dice = -1    
    
    writer = SummaryWriter(os.path.join(args.results_base_dir, args.tb_path))
    print('*')
    print('*')
    print('*')    
    print(f'******** Start training for {epochs} epochs ********')
    print('*')
    print('*')
    print('*')     
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), total = epochs):    
        # training
        model.train()
        for i, (x, y_true, w) in enumerate(loader_train):                        
            x, y_true = x.to(device), y_true.to(device)      
            w = w.to(device)                                            # [Ψ加權]
            n_weighted_epoch += int((w > 1).sum())                      # [Ψ加權]
            y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=label_n_cls).permute(0,3,1,2)  
            y_pred = model(x)           
            optimizer.zero_grad()
            
            if output_channel == 2:
                if label_n_cls == 2:
                    y_true = y_true
                elif label_n_cls == 3:
                    y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                    y_true = y_true[:,0:2,:,:]
                else:
                    raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
            elif output_channel == 3:
                y_true = y_true
            else:
                raise ValueError('Please select correct output channels')
            
            # [Ψ加權] L = Σ w_i (Dice_i + CE_i) / Σ w_i ；w 全為 1 時等於原本的 loss
            l_dice = dice_loss_ps(y_pred, y_true.float()).flatten(1).mean(1)     # (B,)
            l_ce = ce_loss_ps(y_pred, y_true.float()).flatten(1).mean(1)         # (B,)
            #修改 
            loss = (w * (l_dice + l_ce)).mean()                              
            
            if not psi_checked:
                # 第一個 batch 自我檢查：權重全設為 1 時，必須與原本 train.py 的 loss 相同
                with torch.no_grad():
                    loss_orig = dice_loss(y_pred, y_true.float()) + ce_loss(y_pred, y_true.float())
                    loss_w1 = (l_dice + l_ce).mean()
                    d = abs(loss_orig.item() - loss_w1.item())
                print(f'[Ψ加權] 自我檢查：原 loss {loss_orig.item():.8f}　w=1 逐張 loss {loss_w1.item():.8f}　差 {d:.2e}')
                assert d < 1e-5 * max(1.0, abs(loss_orig.item())), '[Ψ加權] 逐張 loss 與原 loss 不一致，停止訓練'
                psi_checked = True
            loss_train.append(loss.item())        
            loss.backward()
            optimizer.step() 
        
        # validation
        model.eval()    
        with torch.no_grad():
            od_dice_metrics = monai.metrics.DiceMetric(include_background=True)
            oc_dice_metrics = monai.metrics.DiceMetric(include_background=True)
            
            for i, (x, y_true) in enumerate(loader_val):                
                x, y_true = x.to(device), y_true.to(device)   
                y_true = F.one_hot(y_true.squeeze(dim=1), num_classes=label_n_cls).permute(0,3,1,2)                  
                y_pred = model(x)
                y_pred = F.interpolate(y_pred, size=(y_true.size(2), y_true.size(3)), mode='bicubic')
                y_pred = torch.argmax(torch.softmax(y_pred,1), dim=1)
                y_pred = F.one_hot(y_pred, num_classes=output_channel).permute(0,3,1,2)
                
                if output_channel == 2:
                    if label_n_cls == 2:
                        od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                        od_dice.append(od_dice_metrics.aggregate().item())
                    elif label_n_cls == 3:
                        y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                        od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                        od_dice.append(od_dice_metrics.aggregate().item())
                    else:
                        raise ValueError('Please select correct number of classes, 2 for OD or 3 for OD/OC')
                elif output_channel == 3:
                    y_true[:,1,:,:] = y_true[:,1,:,:] + y_true[:,2,:,:]
                    y_pred[:,1,:,:] = y_pred[:,1,:,:] + y_pred[:,2,:,:]
                    od_dice_metrics(y_pred[:,1,:,:].unsqueeze(dim=1), y_true[:,1,:,:].unsqueeze(dim=1))
                    oc_dice_metrics(y_pred[:,2,:,:].unsqueeze(dim=1), y_true[:,2,:,:].unsqueeze(dim=1))                                       
                    od_dice.append(od_dice_metrics.aggregate().item())
                    oc_dice.append(oc_dice_metrics.aggregate().item())
                else:
                    raise ValueError('Please select correct output channels')
        
        # record current learning rate, training loss and validation dice, and save the best weights
        lr_current= args.lr
        lr_all.append(lr_current)
        print(f'Epoch_{(epoch+1)} learning rate: {lr_current}')        
        
        loss_epoch_train = np.mean(loss_train)
        loss_train_all.append(loss_epoch_train)
        print(f'Epoch_{(epoch+1)} training loss: {loss_epoch_train}')
        print(f'Epoch_{(epoch+1)} weighted samples: {n_weighted_epoch}')     # [Ψ加權]
        n_weighted_epoch = 0
        loss_train = []  

        if output_channel == 2:
            od_dice_epoch = np.mean(od_dice)
            od_dice_all.append(od_dice_epoch)
            print(f'Epoch_{(epoch+1)} validation OD Dice: {od_dice_epoch}')
            od_dice = []
            
            # save the best model with the most validation dice
            if od_dice_epoch > max_dice:       
                max_dice = od_dice_epoch
                best_epoch = epoch + 1
                best_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                }        
                torch.save(best_checkpoint, os.path.join(args.results_base_dir, args.weights_dir, 'best_weights.pth'))
            
            # tensorboard visualization
            writer.add_scalars('Training Loss and validation dice', {'Loss':loss_epoch_train, 'OD dice':od_dice_epoch}, epoch)
            
        elif output_channel == 3:
            od_dice_epoch = np.mean(od_dice)
            od_dice_all.append(od_dice_epoch)
            print(f'Epoch_{(epoch+1)} validation OD Dice: {od_dice_epoch}')            
            od_dice = []       
            
            oc_dice_epoch = np.mean(oc_dice)
            oc_dice_all.append(oc_dice_epoch)
            print(f'Epoch_{(epoch+1)} validation OC Dice: {oc_dice_epoch}')            
            oc_dice = []   
            
            dice_mean = np.mean([od_dice_epoch, oc_dice_epoch])
            print(f'Epoch_{(epoch+1)} validation mean Dice: {dice_mean}')
            dice_mean_all.append(dice_mean)
            
            # save the best model with the most mean validation dice            
            if dice_mean > max_dice:       
                max_dice = dice_mean
                best_epoch = epoch + 1
                best_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_epoch,
                }        
                torch.save(best_checkpoint, os.path.join(args.results_base_dir, args.weights_dir, 'best_weights.pth'))
            
            # tensorboard visualization
            writer.add_scalars('Training Loss and validation dice', {'Loss':loss_epoch_train, 'OD dice':od_dice_epoch, 'OC dice':oc_dice_epoch, 'Mean dice':dice_mean}, epoch)
        else:
            raise ValueError('Please select correct output channels')       
    writer.close()        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time is {total_time_str}')
    print(f'The best weights are trained in epoch {best_epoch}')

    # training loss and validation dice records visualization
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2    
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    line1 = ax1.plot(range(0, epochs), loss_train_all, label='Training loss', color='#EF5350', linewidth=2, alpha=0.8)[0]
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold', color='#EF5350')
    ax1.tick_params(axis='y', labelcolor='#EF5350')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2 = ax1.twinx()
    lines = [line1]
    if output_channel == 2:   
        line2 = ax2.plot(range(0, epochs), od_dice_all, label='OD dice', color='#4CAF50', linewidth=2, alpha=0.8)[0]
        lines.append(line2)
    elif output_channel == 3:    
        line2 = ax2.plot(range(0, epochs), od_dice_all, label='OD dice', color='#FFEB3B', linewidth=2, alpha=0.8)[0]
        line3 = ax2.plot(range(0, epochs), oc_dice_all, label='OC dice', color='#2196F3', linewidth=2, alpha=0.8)[0]
        line4 = ax2.plot(range(0, epochs), dice_mean_all, label='Mean dice', color='#4CAF50', linewidth=2, alpha=0.8)[0]
        lines.extend([line2, line3, line4])
    ax2.set_ylabel('Dice', fontsize=14, fontweight='bold', color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')
    plt.title('Training records: Loss and Dice', fontsize=16, fontweight='bold', pad=20)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=12, framealpha=0.9)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax1.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax2.ticklabel_format(axis='both', style='plain', useOffset=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_base_dir, args.records_figure_path), dpi=1200)
    plt.show()
    plt.close()

    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
