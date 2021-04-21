#!/usr/bin/env python

import argparse
from time import time
import math
import shutil
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from src import cct as cct_models
from utils.losses import LabelSmoothingCrossEntropy,CxnUnwrapCrossEntropy
from utils.cxnData import cxnDataset

model_names = sorted(name for name in cct_models.__dict__
                     if name.islower() and not name.startswith("_")
                     and callable(cct_models.__dict__[name]))
global best_acc1
best_acc1 = 0


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset') 
    # metavar它为帮助消息中的可选参数提供了不同的名称

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=5, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-2, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--clip-grad-norm', default=0., type=float,
                        help='gradient norm clipping (default: 0 (disabled))')

    parser.add_argument('-m', '--model',
                        type=str.lower,
                        choices=model_names,
                        default='cct_2', dest='model')

    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument('--conv-layers', default=2, type=int,
                        help='number of convolutional layers (cct only)')

    parser.add_argument('--conv-size', default=3, type=int,
                        help='convolution kernel size (cct only)')

    parser.add_argument('--patch-size', default=4, type=int,
                        help='image patch size (vit and cvt only)')

    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')

    parser.add_argument('--disable-aug', action='store_true',
                        help='disable augmentation policies for training')

    parser.add_argument('--gpu-id', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable cuda')    # 只要运行时该变量有传参就将该变量设为True

    return parser


class Args_cxn():
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        self.workers = 2
        self.data = 'DIR'
        self.print_freq = 1
        self.checkpoint_path = 'checkpoint.pth'
        self.epochs = 10
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
        

def plot_3d_wrap(image_t,image_true,image_wrap):
    image_wrap=image_wrap[0,0,:,:]
    image_wrap = image_wrap.detach().numpy()
    image_true=image_true[0,0,:,:]
    image_true = image_true.detach().numpy()
    image_t = image_t[0,0,:,:]
    image_t = image_t.detach().numpy()
    N = 256
    X = np.arange(-3,3,6/N)
    Y = np.arange(-3,3,6/N)
    X,Y=np.meshgrid(X,Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.subplot(211)
    ax.plot_surface(X, Y, image_t, rstride=1, cstride=1, antialiased=True)
    plt.subplot(222)
    ax.plot_surface(X, Y, image_true, rstride=1, cstride=1, antialiased=True)
    plt.subplot(221)
    ax.plot_surface(X, Y, image_wrap, rstride=1, cstride=1, antialiased=True)
    plt.show()
    
    
def imagesc(image_t,image_true,image_wrap):
    image_wrap=image_wrap[0,0,:,:]
    image_wrap = image_wrap.detach().numpy()
    image_true=image_true[0,0,:,:]
    image_true = image_true.detach().numpy()
    image_t = image_t[0,0,:,:]
    image_t = image_t.detach().numpy()
    plt.axis('on')
    plt.subplot(131)
    plt.imshow(image_t)
    plt.colorbar(shrink=0.4)
    plt.subplot(132)
    plt.imshow(image_true)
    plt.colorbar(shrink=0.4)
    plt.subplot(133)
    plt.imshow(image_wrap)
    plt.colorbar(shrink=0.4)
    plt.show()
    

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def cls_train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()   # 开启模型的训练模式
    loss_val = 0
    n = 0
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu_id, non_blocking=True)
            target = target.cuda(args.gpu_id, non_blocking=True)
        output = model(images)
        target = target[:,0,:,:].unsqueeze(1)  # unsqueeze(1)增加个第1维
        loss = criterion(output, target)

        # acc1 = accuracy(output, target)
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        # acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()   # 首先要清空优化器的梯度,只算这次怎么优化
        loss.backward()         # 反向传播

        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        optimizer.step()        # 梯度做进一步参数更新

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss = (loss_val / n)
            print(f'[Epoch {epoch+1}][Train][{i}] \t Loss: {avg_loss:.4e} ')
    imagesc(output,target,images)


def cls_validate(val_loader, model, criterion, args, epoch=None, time_begin=None):
    model.eval()    # 开启模型的测试模式,此模式不dropout也不backwards
    loss_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu_id, non_blocking=True)
                target = target.cuda(args.gpu_id, non_blocking=True)

            output = model(images)
            target = target[:,0,:,:].unsqueeze(1)
            loss = criterion(output, target)
            # acc1 = accuracy(output, target)
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            # acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss = (loss_val / n)
                print(f'[Epoch {epoch+1}][Eval][{i}] \t Loss: {avg_loss:.4e}')

    avg_loss = (loss_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch+1}] \t \t  Time: {total_mins:.2f}')

    return avg_loss


if __name__ == '__main__':
    args = Args_cxn()
    img_size = 256
    img_mean, img_std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

    model = cct_models.__dict__[args.model](img_size=img_size,
                                        positional_embedding=args.positional_embedding,
                                        n_conv_layers=args.conv_layers,
                                        kernel_size=args.conv_size,
                                        patch_size=args.patch_size)

    criterion = nn.MSELoss()    # 这里也是要改的,用原来的就可以
    

    if (not args.no_cuda) and torch.cuda.is_available():
          torch.cuda.set_device(args.gpu_id)
          model.cuda(args.gpu_id)
          criterion = criterion.cuda(args.gpu_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    '''
    augmentations = []
    if not False:
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
        ]
    augmentations += [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]

    augmentations = transforms.Compose(augmentations)
    train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True,
                                      transform=augmentations)

    val_dataset = datasets.CIFAR10(
        root=args.data, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))
    '''
    
    train_dataset = cxnDataset('./trainx/','./trainy/')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_dataset = cxnDataset('./valx/','./valy/')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    print("Beginning training")
    time_begin = time()
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        cls_train(train_loader, model, criterion, optimizer, epoch, args)
        acc1 = cls_validate(val_loader, model, criterion, args, epoch=epoch, 
                            time_begin=time_begin)
        best_acc1 = min(acc1, best_acc1)
        torch.save(model.state_dict(), args.checkpoint_path)  # 每个epoch都要存

    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_acc1:.2f}, '
          f'final top-1: {acc1:.2f}')
    
