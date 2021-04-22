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


class Args_cxn():
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        self.workers = 2
        self.data = 'DIR'
        self.print_freq = 1
        self.checkpoint_path='./drive/MyDrive/transformer_unwrap/checkpoint.pth'
        self.epochs = 100
        self.warmup = 5
        self.batch_size = 32
        self.lr = 0.0005
        self.weight_decay = 3e-2
        self.clip_grad_norm = 10
        self.model = 'cct_2'
        self.positional_embedding = 'learnable' # choices=['learnable', 'sine', 'none']
        self.conv_layers = 2
        self.conv_size = 3
        self.patch_size = 4
        self.disable_cos = False
        self.disable_aug = False
        self.gpu_id = 0
        self.no_cuda = False
        self.add_all_features = True
        

img_size = 256
args = Args_cxn()
model = cct_models.__dict__[args.model](img_size=img_size,
                                        positional_embedding=args.positional_embedding,
                                        n_conv_layers=args.conv_layers,
                                        kernel_size=args.conv_size,
                                        patch_size=args.patch_size,
                                        add_all_features=args.add_all_features)
checkpoint_path='./drive/MyDrive/transformer_unwrap/checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

