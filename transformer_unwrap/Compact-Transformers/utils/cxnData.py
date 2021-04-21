import torch,os,random,time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def generate_img():
    if not os.path.exists('./trainx'):
      os.makedirs('./trainx')
    if not os.path.exists('./trainy'):
      os.makedirs('./trainy')  
    if not os.path.exists('./valx'):
      os.makedirs('./valx')
    if not os.path.exists('./valy'):
      os.makedirs('./valy')
    N = 512
    X = np.arange(-3,3,6/N)
    Y = np.arange(-3,3,6/N)
    X,Y=np.meshgrid(X,Y)
    for i in range(10): # 每种模式进行10次计算,
      t=time.time()
      file_first_name = str(t*1000000)
      image_t = 20*np.exp(-0.25*(X**2 + Y**2)) + 2*X*np.random.rand(1) + Y*np.random.rand(1)
      image_val = X * np.exp(-X**2-X**2) + X*np.random.rand(1) + Y*np.random.rand(1)

      for noise_variance in np.arange(0,0.4,0.01):
        image_noise = image_t+noise_variance*np.random.randn(N,N)
        image_val_noise = image_val+noise_variance*np.random.randn(N,N)
        imagen_wrapped = np.arctan2(np.sin(image_noise), np.cos(image_noise))
        image_val_wrapped = np.arctan2(np.sin(image_val_noise), np.cos(image_val_noise))
        np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(imagen_wrapped))
        np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_t))
        np.save('./valx/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_val_wrapped))
        np.save('./valy/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_val))


class cxnDataset(Dataset):
    """
     image_files：wrap存放地址根路径
     label_files：unwrap存放地址根路径
     augment：是否需要图像增强
     return: tuple: (x,y)
    """
    def __init__(self, trainx_root, trainy_root, augment=None):
        # 这个list存放所有图像的地址
        self.image_files = np.array([x.path for x in os.scandir(trainx_root)
                                     if x.name.endswith(".npy") or
                                     x.name.endswith(".png") or 
                                     x.name.endswith(".JPG")])
        self.label_files = np.array([x.path for x in os.scandir(trainy_root)
                                     if x.name.endswith(".npy") or
                                     x.name.endswith(".png") or 
                                     x.name.endswith(".JPG")])
        self.augment = augment   # 是否需要图像增强
        

    def __getitem__(self, index):
        if self.augment:
          image = self.open_image(self.image_files[index])
          image = self.augment(image)  # 这里对图像进行了增强
          return self.to_tensor(image)      # 将读取到的图像变成tensor再传出
        else:
          # 如果不进行增强，直接读取图像数据并返回
          # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
          img = self.to_tensor(self.open_image(self.image_files[index]))
          label = self.to_tensor(self.open_image(self.label_files[index]))
          return (img,label)    # 返回tuple形式的训练集

    def open_image(self,name):
        img = np.load(name)
        return img
        
    
    def to_tensor(self,img):
        # unsqueeze(0)在第一维上增加一个维度
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        img = img.repeat(3,1,1)
        return img
        

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)
    

# cxn = cxnDataset('./trainx/','./trainy/')
# train_loader = torch.utils.data.DataLoader(
#       cxn, batch_size=32, shuffle=True,
#       num_workers=2)


