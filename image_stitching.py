'''
author: cxn
version: 0.1.0
图像拼接
'''

import numpy as np
import argparse
import cv2, os, glob
import matplotlib.pyplot as plt

def stitchs():
    images = []
     
    # loop over the image paths, load each one, and add them to our images to stitch list
    path = os.getcwd() + '\\origin_image'
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        image = cv2.imread(file_path)
        images.append(image)
        
    stitcher = cv2.createStitcher()
    (status, stitched) = stitcher.stitch(images)
    # b,g,r = cv2.split(stitched) #分离颜色数组
    # img2 = cv2.merge([r,g,b]) #按Rgb进行合并
    if status == 0:
        plt.imshow(stitched)
    else:
    	print("[INFO] image stitching failed ({})".format(status))
        

def sift_transform():
    path = os.getcwd() + '\\origin_image'
    sift = cv2.xfeatures2d.SIFT_create()
    imgname1 = path+'\\origin.png'
    gray1 = cv2.imread(imgname1, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift.detectAndCompute(gray1,None)   #des是描述子
    
    imgname2 = path+'\\40_1.bmp'
    gray2 = cv2.imread(imgname2, cv2.IMREAD_GRAYSCALE)
    kp2, des2 = sift.detectAndCompute(gray2,None)  #des是描述子
    
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    
    img_out = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:20], None, flags=2)
    plt.imshow(img_out)
    

def my_calibrate(x_nums = 11,y_nums = 11, chess_len = 10):
    """
    进行x,y,z计算时,我需要用两台相机来测其三个方向的坐标,其中用来测一个方向坐标时,
    尽可能使标定板平面与相机垂直

    """
    # x,y方向上的角点个数
    # 设置(生成)标定图在世界坐标中的坐标
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)  
    # 生成x_nums*y_nums个坐标,每个坐标包含x,y,z三个元素
    world_point[:, :2]=chess_len * np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)
    # mgrid[]生成包含两个二维矩阵的矩阵,每个矩阵都有x_nums列,y_nums行,
    # 我这里用的是10mm×10mm的方格,所以乘了10,以mm代表世界坐标的计量单位
    world_position = [] #存放世界坐标
    image_position = [] #存放棋盘角点对应的图片像素坐标
    # 设置世界坐标的坐标
    axis = chess_len * np.float32(
        [[3, 0, 0], [0, 3, 0], [1, 0, 0]]).reshape(-1, 3)  
    # axis列数为3表示xyz,行数不知,表示要画几个点
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 获取所有标定图
    photo_path = os.getcwd() + '\\cali_img'
    image_paths = glob.glob(photo_path + '\\*.bmp')  # 遍历文件并保存列表
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 查找角点
        ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), None)
        """
    		如果能找得到角点:返回角点对应的像素坐标,并且将其对应到世界坐标中
    		世界坐标[0,0,0],[0,1,0].....
    		图像坐标[10.123123,20.123122335],[19.123123,21.123123123]....
        """
        if ok:
            # 把每一幅图像的世界坐标放到world_position中
            world_position.append(world_point)
            # 获取更精确的角点位置
            exact_corners = cv2.cornerSubPix(gray, corners, 
                                             (11, 11), (-1, -1), criteria)
            # 把获取的角点坐标放到image_position中
            image_position.append(exact_corners)
            # 可视化角点
            image = cv2.drawChessboardCorners(image,(x_nums,y_nums),
                                              exact_corners,ok)
            plt.imshow(image)
            plt.show()  # 相当于matlab的hold on
            
    """
    点对应好了,开始计算内参,畸变矩阵,外参
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        world_position, image_position, gray.shape[::-1], None, None)
    # 内参是mtx,畸变矩阵是dist,旋转向量(要得到矩阵还要进行罗德里格斯变换)rvecs,
    # 外参:平移矩阵tvecs
    # 将内参保存起来
    # 获取第一幅图象的外参
    rotation_matrix = []    # 用它左乘世界坐标得相机坐标
    for i in range(len(image_position)):
        _, rvec0, tvec0, inliers0 = cv2.solvePnPRansac(
            world_position[i], image_position[i], mtx, dist)
        rotation_m, _ = cv2.Rodrigues(rvec0) # 罗德里格斯变换成3x3的矩阵
        rotation_t = np.hstack([rotation_m,tvec0])
        rotation_t_Homogeneous_matrix = np.vstack(
            [rotation_t,np.array([[0, 0, 0, 1]])])  # 用它左乘世界坐标得相机坐标
        rotation_matrix.append(rotation_t_Homogeneous_matrix)
        # 函数projectPoints()根据所给的3D坐标和已知的几何变换来求解投影后的2D坐标
        imgpts, jac = cv2.projectPoints(axis, rvec0, tvec0, mtx, dist)
        imageht = cv2.imread(image_paths[i])
        imagehuatu = cv2.drawChessboardCorners(imageht,(3,3),
                                              imgpts,ok)
        plt.imshow(imagehuatu)
        plt.show()
        
    np.savez(os.getcwd() + '\\internal_reference\\internal_reference',
             mtx=mtx, dist=dist)
    
    print('内参是:\n', mtx, '\n畸变参数是:\n', dist,
       '\n外参:旋转向量(要得到矩阵还要进行罗德里格斯变换)是:\n',
       rvecs, '\n外参:平移矩阵是:\n',tvecs)
          
    # 计算偏差
    mean_error = 0
    for i in range(len(world_position)):
        image_position2, _ = cv2.projectPoints(world_position[i], rvecs[i], 
                                               tvecs[i], mtx, dist)
        error = cv2.norm(image_position[i], image_position2, 
                         cv2.NORM_L2) / len(image_position2)
        mean_error += error
    print("total error: ", mean_error / len(image_position))
    
    
my_calibrate()   
    