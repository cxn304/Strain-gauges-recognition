import torch,os,random,time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def test_generate_img():
    N = 256
    X = np.arange(-3,3,6/N)
    Y = np.arange(-3,3,6/N)
    X,Y=np.meshgrid(X,Y)
    
    unwraped = []
    wrapped = []
    '''
    image_t = random.randint(15,20)*np.exp(-0.1*(X**2 + Y**2)) + X*np.random.rand(1) + Y*np.random.rand(1)
    imagen_wrappedt = np.arctan2(np.sin(image_t), np.cos(image_t))
    unwraped.append(image_t)
    wrapped.append(imagen_wrappedt)
    
    image_t1 = random.randint(15,20)*X * np.exp(-X**2-X**2) + random.randint(5,15)*X*np.random.rand(1) + Y*np.random.rand(1)
    imagen_wrappe_t1 = np.arctan2(np.sin(image_t1), np.cos(image_t1))
    unwraped.append(image_t1)
    wrapped.append(imagen_wrappe_t1)
    '''
    image_t2 = random.randint(5,10)*(1-X/2+X**5+Y**3)*random.randint(5,10)*np.exp(-X ** 2 - Y ** 2)
    imagen_wrappe_t2 = np.arctan2(np.sin(image_t2), np.cos(image_t2))
    unwraped.append(image_t2)
    wrapped.append(imagen_wrappe_t2)
    
    image_t3 = random.randint(5,10)*np.sin(X)+random.randint(5,10)*np.cos(Y)
    imagen_wrappe_t3 = np.arctan2(np.sin(image_t3), np.cos(image_t3))
    unwraped.append(image_t3)
    wrapped.append(imagen_wrappe_t3)
    
    image_t4=random.randint(5,15)*Y*np.sin(X)-random.randint(5,15)*X*np.cos(Y)
    imagen_wrappe_t4 = np.arctan2(np.sin(image_t4), np.cos(image_t4))
    unwraped.append(image_t4)
    wrapped.append(imagen_wrappe_t4)
    
    image_t5=random.randint(5,15)*(1-X)**2*np.exp(-(X**2)-(Y+1)**2)- random.randint(5,15)*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)- 1/3*np.exp(-(X+1)**2 - Y**2)
    imagen_wrappe_t5 = np.arctan2(np.sin(image_t5), np.cos(image_t5))  
    unwraped.append(image_t5)
    wrapped.append(imagen_wrappe_t5)
    
    image_t6=random.randint(5,10)*X
    imagen_wrappe_t6 = np.arctan2(np.sin(image_t6), np.cos(image_t6))  
    unwraped.append(image_t6)
    wrapped.append(imagen_wrappe_t6)
    
    image_t7=random.randint(5,10)*Y
    imagen_wrappe_t7 = np.arctan2(np.sin(image_t7), np.cos(image_t7))  
    unwraped.append(image_t7)
    wrapped.append(imagen_wrappe_t7)
    
    image_t8=random.randint(5,10)*Y+random.randint(5,10)*X
    imagen_wrappe_t8 = np.arctan2(np.sin(image_t8), np.cos(image_t8))  
    unwraped.append(image_t8)
    wrapped.append(imagen_wrappe_t8)
    
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace =.5, hspace =.5) # 调整子图间距
  
    for i in range(len(wrapped)):
        ax = plt.subplot(5,4,2*i+1)
        plt.imshow(unwraped[i])
        plt.colorbar(shrink=0.9)
        ax.set_title('unwrap Mat')
        ax = plt.subplot(5,4,2*i+2)
        plt.imshow(wrapped[i])
        plt.colorbar(shrink=0.9)
        ax.set_title('wrapped Mat')



def generate_img():
    '''
    generate noise image
    '''
    if not os.path.exists('./trainx'):
      os.makedirs('./trainx')
    if not os.path.exists('./trainy'):
      os.makedirs('./trainy')  
    if not os.path.exists('./valx'):
      os.makedirs('./valx')
    if not os.path.exists('./valy'):
      os.makedirs('./valy')
    N = 256
    X = np.arange(-3,3,6/N)
    Y = np.arange(-3,3,6/N)
    X,Y=np.meshgrid(X,Y)
    for i in range(2,16,1): # 每种模式进行10次计算
        t=time.time()
        file_first_name = str(t*1000000)
        image_t = i**np.exp(-0.25*(X**2 + Y**2)) + 2*X*np.random.rand(1) + Y*np.random.rand(1)
        image_t1 = i*X * np.exp(-X**2-X**2) + i*X*np.random.rand(1) + Y*np.random.rand(1)
        image_t2 = i*(1-X/2+X**5+Y**3)*i*np.exp(-X ** 2 - Y ** 2)
        image_t3 = i*X*np.exp(-X**2-Y**2)
        image_t4=i*Y*np.sin(X)-i*X*np.cos(Y)
        image_t5=i*(1-X)**2*np.exp(-(X**2)-(Y+1)**2)-i*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)- 1/3*np.exp(-(X+1)**2 - Y**2)
        image_t6=i*X
        image_t7=i*Y
        image_t8=i*Y+i*X
        for noise_variance in np.arange(0,0.4,0.01):
            image_noise = image_t+noise_variance*np.random.randn(N,N)
            image_t_wrapped = np.arctan2(np.sin(image_noise), np.cos(image_noise))
            
            image_t1_noise = image_t1+noise_variance*np.random.randn(N,N)
            image_t1_wrapped = np.arctan2(np.sin(image_t1_noise), 
                                          np.cos(image_t1_noise))
            
            image_t2_noise = image_t2+noise_variance*np.random.randn(N,N)
            image_t2_wrapped = np.arctan2(np.sin(image_t2_noise), 
                                          np.cos(image_t2_noise))
            
            image_t3_noise = image_t3+noise_variance*np.random.randn(N,N)
            image_t3_wrapped = np.arctan2(np.sin(image_t3_noise), 
                                          np.cos(image_t3_noise))
            
            image_t4_noise = image_t4+noise_variance*np.random.randn(N,N)
            image_t4_wrapped = np.arctan2(np.sin(image_t4_noise), 
                                          np.cos(image_t4_noise))
            
            image_t5_noise = image_t5+noise_variance*np.random.randn(N,N)
            image_t5_wrapped = np.arctan2(np.sin(image_t5_noise), 
                                          np.cos(image_t5_noise))
            
            image_t6_noise = image_t6+noise_variance*np.random.randn(N,N)
            image_t6_wrapped = np.arctan2(np.sin(image_t6_noise), 
                                          np.cos(image_t6_noise))
            
            image_t7_noise = image_t7+noise_variance*np.random.randn(N,N)
            image_t7_wrapped = np.arctan2(np.sin(image_t7_noise), 
                                          np.cos(image_t7_noise))
            
            image_t8_noise = image_t8+noise_variance*np.random.randn(N,N)
            image_t8_wrapped = np.arctan2(np.sin(image_t8_noise), 
                                          np.cos(image_t8_noise))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'.npy',np.float32(image_t_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'.npy',np.float32(image_t))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t1.npy',np.float32(image_t1_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t1.npy',np.float32(image_t1))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t2.npy',np.float32(image_t2_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t2.npy',np.float32(image_t2))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t3.npy',np.float32(image_t3_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t3.npy',np.float32(image_t3))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t4.npy',np.float32(image_t4_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t4.npy',np.float32(image_t4))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t5.npy',np.float32(image_t5_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t5.npy',np.float32(image_t5))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t6.npy',np.float32(image_t6_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t6.npy',np.float32(image_t6))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t7.npy',np.float32(image_t7_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t7.npy',np.float32(image_t7))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t8.npy',np.float32(image_t8_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t8.npy',np.float32(image_t8))
            
            np.save('./valx/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_t1_wrapped))
            np.save('./valy/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_t1))


def generate_aliasing_img():
    '''
    generate aliasing image
    '''
    unwraped = []
    wrapped = []
    if not os.path.exists('./trainx'):
      os.makedirs('./trainx')
    if not os.path.exists('./trainy'):
      os.makedirs('./trainy')  
    if not os.path.exists('./valx'):
      os.makedirs('./valx')
    if not os.path.exists('./valy'):
      os.makedirs('./valy')
    N = 256
    X = np.arange(-3,3,6/N)
    Y = np.arange(-3,3,6/N)
    X,Y=np.meshgrid(X,Y)
    
    image_t2 = random.randint(5,10)*(1-X/2+X**5+Y**3)*random.randint(5,10)*np.exp(-X ** 2 - Y ** 2)
    imagen_wrappe_t2 = np.arctan2(np.sin(image_t2), np.cos(image_t2))
    unwraped.append(image_t2)
    wrapped.append(imagen_wrappe_t2)


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
    
    
# test_generate_img()
# generate_img()
# cxn = cxnDataset('./trainx/','./trainy/')
# train_loader = torch.utils.data.DataLoader(
#       cxn, batch_size=32, shuffle=True,
#       num_workers=2)


