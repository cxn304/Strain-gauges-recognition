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
import os,shutil,pdb
import scipy.io as io


def imagesc(image_t1,image_true1,image_wrap1,args):
    jiange = 1
    plt.figure(figsize=(17, 12))
    plt.subplots_adjust(wspace =.4, hspace =.4) # 调整子图间距
    plt.axis('on')
    predict = []
    trues = []
    inputs = []
    error_2d = []
    for i in range(0,4,jiange):
        image_wrap=image_wrap1[i,0,:,:]
        if (not args.no_cuda) and torch.cuda.is_available():
            image_wrap = image_wrap.detach().cpu().numpy()
        else:
            image_wrap = image_wrap.detach().numpy()
        inputs.append(image_wrap)
        image_true=image_true1[i,0,:,:]
        if (not args.no_cuda) and torch.cuda.is_available():
            image_true = image_true.detach().cpu().numpy()
        else:
            image_true = image_true.detach().numpy()
        trues.append(image_true)
        image_t = image_t1[i,0,:,:]
        if (not args.no_cuda) and torch.cuda.is_available():
            image_t = image_t.detach().cpu().numpy()
        else:
            image_t = image_t.detach().numpy()
        predict.append(image_t)
        error_2d.append(image_t-image_true)
    
    xx = np.arange((predict[0].shape)[0])
    for i in range(len(trues)):
        ax = plt.subplot(4,5,5*i+1)
        plt.imshow(predict[i])
        plt.colorbar(shrink=0.6)
        ax.set_title('Unwrap Mat Predict')
        ax = plt.subplot(4,5,5*i+2)
        plt.imshow(trues[i])
        plt.colorbar(shrink=0.6)
        ax.set_title('Unwrap Mat True')
        ax = plt.subplot(4,5,5*i+3)
        plt.imshow(inputs[i])
        plt.colorbar(shrink=0.6)
        ax.set_title('Wraped Mat Input')
        ax = plt.subplot(4,5,5*i+4)
        plt.ylabel('phase')
        plt.xlabel('col')
        plt.plot(xx, trues[i][len(xx)//2,:], color='green',
                 label='True Unwrap')
        plt.plot(xx, predict[i][len(xx)//2,:], color='red', 
                 label='Predict Unwrap')
        plt.legend()
        ax.set_title('Result of row 128')
        ax = plt.subplot(4,5,5*i+5)
        plt.imshow(error_2d[i],vmin=-1, vmax=1)
        plt.colorbar(shrink=0.6)
        ax.set_title('Full field error')
    plt.show()
    plt.close()
   
    

class Args_cxn():
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        self.workers = 2
        self.data = 'DIR'
        self.print_freq = 5
        self.model = "cot_2"
        self.checkpoint_path=\
            "./drive/MyDrive/transformer_unwrap/ucxncot2/ucot_2_t5.pth"
        self.epochs = 200
        self.warmup = 5
        self.batch_size = 5
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
        

def tested():
    img_size = 1024
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
              
    '''
    path_checkpoint = args.checkpoint_path
    if torch.cuda.is_available():
      checkpoint = torch.load(path_checkpoint)
    else:
      checkpoint = torch.load(path_checkpoint,map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    '''
    
    train_dataset = cxnDataset('./train_wrapped/','./train_unwrap/')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)   # 没有名字只有数据
    # model.eval()
    for i, (images, target) in enumerate(train_loader):
        if i == 32:
            break
        imagesc(images,target,images,args)
        '''
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu_id, non_blocking=True)
            target = target.cuda(args.gpu_id, non_blocking=True)
        output = model(images)
        target = target[:,0,:,:].unsqueeze(1)  # unsqueeze(1)增加个第1维
        imagesc(output,target,images,args)
        '''


def generate_roudian_img():
    if not os.path.exists('./train_wrapped'):
      os.makedirs('./train_wrapped')
    if not os.path.exists('./train_unwrap'):
      os.makedirs('./train_unwrap')
    wrappedDir = "./train_wrapped/"
    unwrapDir = "./train_unwrap/"
    originData = os.listdir("./roudianTrueImage/")
    for dMoveUpDown in range(10,200,10):
      for dMoveLeftRight in range(10,200,10):
        for tmpName in originData:
          matPath = "./roudianTrueImage/" + tmpName
          tmpName = tmpName[:-4]
          if tmpName[0] == 'u':
            image_now = io.loadmat(matPath)['unwrap']
            saveImageDir = unwrapDir+tmpName+'_'+str(dMoveUpDown)+'_'+str(dMoveLeftRight)+'.npy'
          else:
            image_now = io.loadmat(matPath)['wrapped']
            saveImageDir = wrappedDir+tmpName+'_'+str(dMoveUpDown)+'_'+str(dMoveLeftRight)+'.npy'
          image_new = np.zeros(image_now.shape)
          image_new[:-dMoveUpDown,:-dMoveLeftRight]=image_now[dMoveUpDown-1:-1,dMoveLeftRight-1:-1]
          image_new[np.isnan(image_new)]=0
          plt.imshow(image_new)
          plt.figure()
          plt.plot(np.arange(len(image_new)),image_new[len(image_new)//2,:])
          plt.figure()
          plt.imshow(image_new[::2,::2])
          plt.figure()
          plt.plot(np.arange(len(image_new[::2,::2])),image_new[::2,::2][len(image_new[::2,::2])//2,:])
          plt.figure()
          plt.imshow(image_new[::4,::4])
          plt.figure()
          plt.plot(np.arange(len(image_new[::4,::4])),image_new[::4,::4][len(image_new[::4,::4])//2,:])
          plt.show()
          pdb.set_trace()
          #np.save(saveImageDir,np.array(image_new, dtype="float32"))
          
        
# generate_roudian_img()

if __name__ == '__main__':
    tested()


