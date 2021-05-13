import os,random,time
import numpy as np
import scipy.io as io


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
    for i in np.arange(9,11,1): # 每种模式进行10次计算
        t=time.time()
        file_first_name = str(t*1000000)
        # image_t = i**np.exp(-0.25*(X**2 + Y**2)) + 2*X*np.random.rand(1) + Y*np.random.rand(1)
        # image_t1 = i*X * np.exp(-X**2-X**2) + i*X*np.random.rand(1) + Y*np.random.rand(1)
        # image_t2 = i*(1-X/2+X**5+Y**3)*i*np.exp(-X ** 2 - Y ** 2)
        # image_t3 = i*np.sin(X)+i*np.cos(Y)
        # image_t4=i*Y*np.sin(X)-i*X*np.cos(Y)
        # image_t5=i*(1-X)**2*np.exp(-(X**2)-(Y+1)**2)-i*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)- 1/3*np.exp(-(X+1)**2 - Y**2)
        # image_t6=i*X
        image_t7=i*Y
        # image_t8=i*Y+i*X
        for noise_variance in np.arange(0,0.4,0.01):
            # image_noise = image_t+noise_variance*np.random.randn(N,N)
            # image_t_wrapped = np.arctan2(np.sin(image_noise), np.cos(image_noise))
            
            # image_t1_noise = image_t1+noise_variance*np.random.randn(N,N)
            # image_t1_wrapped = np.arctan2(np.sin(image_t1_noise), 
            #                               np.cos(image_t1_noise))
            
            # image_t2_noise = image_t2+noise_variance*np.random.randn(N,N)
            # image_t2_wrapped = np.arctan2(np.sin(image_t2_noise), 
            #                               np.cos(image_t2_noise))
            
            # image_t3_noise = image_t3+noise_variance*np.random.randn(N,N)
            # image_t3_wrapped = np.arctan2(np.sin(image_t3_noise), 
                                          # np.cos(image_t3_noise))
            
            # image_t4_noise = image_t4+noise_variance*np.random.randn(N,N)
            # image_t4_wrapped = np.arctan2(np.sin(image_t4_noise), 
            #                               np.cos(image_t4_noise))
            
            # image_t5_noise = image_t5+noise_variance*np.random.randn(N,N)
            # image_t5_wrapped = np.arctan2(np.sin(image_t5_noise), 
            #                               np.cos(image_t5_noise))
            
            # image_t6_noise = image_t6+noise_variance*np.random.randn(N,N)
            # image_t6_wrapped = np.arctan2(np.sin(image_t6_noise), 
            #                               np.cos(image_t6_noise))
            
            image_t7_noise = image_t7+noise_variance*np.random.randn(N,N)
            image_t7_wrapped = np.arctan2(np.sin(image_t7_noise), 
                                          np.cos(image_t7_noise))
            
            # image_t8_noise = image_t8+noise_variance*np.random.randn(N,N)
            # image_t8_wrapped = np.arctan2(np.sin(image_t8_noise), 
            #                               np.cos(image_t8_noise))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'.npy',np.float32(image_t_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'.npy',np.float32(image_t))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t1.npy',np.float32(image_t1_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t1.npy',np.float32(image_t1))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t2.npy',np.float32(image_t2_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t2.npy',np.float32(image_t2))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t3.npy',np.float32(image_t3_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t3.npy',np.float32(image_t3))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t4.npy',np.float32(image_t4_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t4.npy',np.float32(image_t4))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t5.npy',np.float32(image_t5_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t5.npy',np.float32(image_t5))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t6.npy',np.float32(image_t6_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t6.npy',np.float32(image_t6))
            
            np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t7.npy',np.float32(image_t7_wrapped))
            np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
                round(noise_variance,2))+'t7.npy',np.float32(image_t7))
            
            # np.save('./trainx/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t8.npy',np.float32(image_t8_wrapped))
            # np.save('./trainy/'+file_first_name+'_'+str(i)+'_'+str(
            #     round(noise_variance,2))+'t8.npy',np.float32(image_t8))
            
            # np.save('./valx/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_t1_wrapped))
            # np.save('./valy/'+file_first_name+'_'+str(i)+'_'+str(round(noise_variance,2))+'.npy',np.float32(image_t1))


def npy_mat(npy_path,mat_path):
    if not os.path.exists(mat_path):
      os.makedirs(mat_path)
    npyname_path = os.listdir(npy_path)
    for npyname in npyname_path:
        name = npyname[:-4]
        mat_name = name+'.mat'
        mat_name = os.path.join(mat_path,mat_name)
        npy = np.load(npy_path+'/'+npyname)
        io.savemat(mat_name,{'data':npy})
        
# generate_img()
# npy_mat('./trainx','./trainx_mat')
# npy_mat('./trainy','./trainy_mat')
