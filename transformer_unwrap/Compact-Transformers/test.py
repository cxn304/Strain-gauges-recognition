'''
author: cxn
version: 0.1.0
unwrap test
'''

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src import cct as cct_models
from utils.cxnData import cxnDataset
from main import imagesc


class Args_cxn():
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        self.workers = 2
        self.data = 'DIR'
        self.print_freq = 5
        self.model = "cct_6"
        self.checkpoint_path=\
            "./drive/MyDrive/transformer_unwrap/before_cct6/contact_before_cct_6.pth"
        self.epochs = 200
        self.warmup = 5
        self.batch_size = 128
        self.lr = 0.0005
        self.weight_decay = 3e-2
        self.clip_grad_norm = 10
        self.positional_embedding = 'learnable' # choices=['learnable', 'sine', 'none']
        self.conv_layers = 2
        self.conv_size = 3
        self.patch_size = 4
        self.disable_cos = False
        self.disable_aug = False
        self.gpu_id = 0
        self.no_cuda = False
        self.add_all_features = False   # 是否在解码器中添加
        self.RESUME = False
        

img_size = 256
args = Args_cxn()
model = cct_models.__dict__[args.model](img_size=img_size,
                                        positional_embedding=args.positional_embedding,
                                        n_conv_layers=args.conv_layers,
                                        kernel_size=args.conv_size,
                                        patch_size=args.patch_size,
                                        add_all_features=args.add_all_features)

if (not args.no_cuda) and torch.cuda.is_available():
          torch.cuda.set_device(args.gpu_id)
          model.cuda(args.gpu_id)
          
          
path_checkpoint = args.checkpoint_path
if torch.cuda.is_available():
  checkpoint = torch.load(path_checkpoint)
else:
  checkpoint = torch.load(path_checkpoint,map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])

train_dataset = cxnDataset('./trainx/','./trainy/')

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers)
model.eval()
for i, (images, target) in enumerate(train_loader):
    if i == 2:
        break
    if (not args.no_cuda) and torch.cuda.is_available():
        images = images.cuda(args.gpu_id, non_blocking=True)
        target = target.cuda(args.gpu_id, non_blocking=True)
    output = model(images)
    target = target[:,0,:,:].unsqueeze(1)  # unsqueeze(1)增加个第1维
    imagesc(output,target,images)

