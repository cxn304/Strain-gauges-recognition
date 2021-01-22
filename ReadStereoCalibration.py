'''
author: cxn
version: 0.1.0
read camera calibration from mat
'''


import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt


#双目相机参数
class stereoCameral(object):
    def __init__(self):
        stereoParameters = loadmat("./internal_reference/stereoParameters.mat")
        self.cam_matrix_left = stereoParameters["stereoParameters"]["K1"][0][0]  # IntrinsicMatrix
        self.distortion_l = stereoParameters["stereoParameters"]["D1"][0][0]  # distortion
        self.cam_matrix_right = stereoParameters["stereoParameters"]["K2"][0][0]
        self.distortion_r = stereoParameters["stereoParameters"]["D2"][0][0]
        self.size = stereoParameters["stereoParameters"]["size"][0][0]  # image size
        self.R = stereoParameters["stereoParameters"]["rot"][0][0].T
        self.T = stereoParameters["stereoParameters"]["trans"][0][0]


def getRectifyTransform(height, width, config):
    #读取矩阵参数
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    #计算校正变换,cv2.stereoRectify
    """
    stereoRectify() 的作用是为每个摄像头计算立体校正的映射矩阵.
    所以其运行结果并不是直接将图片进行立体矫正,而是得出进行立体矫正所需要的映射矩阵
    cameraMatrix1-第一个摄像机的摄像机矩阵
    distCoeffs1-第一个摄像机的畸变向量
    cameraMatrix2-第二个摄像机的摄像机矩阵
    distCoeffs1-第二个摄像机的畸变向量
    imageSize-图像大小
    R- stereoCalibrate() 求得的R矩阵
    T- stereoCalibrate() 求得的T矩阵
    R1-输出矩阵,第一个摄像机的校正变换矩阵（旋转变换）
    R2-输出矩阵,第二个摄像机的校正变换矩阵（旋转矩阵）
    P1-输出矩阵,第一个摄像机在新坐标系下的投影矩阵
    P2-输出矩阵,第二个摄像机在想坐标系下的投影矩阵
    Q-4*4的深度差异映射矩阵
    flags-可选的标志有两种零或者CV_CALIB_ZERO_DISPARITY,
    如果设置 CV_CALIB_ZERO_DISPARITY 的话,该函数会让两幅校正后的图像的主点
    有相同的像素坐标.否则该函数会水平或垂直的移动图像,以使得其有用的范围最大
    alpha-拉伸参数.如果设置为负或忽略,将不进行拉伸.如果设置为0,那么校正后图像
    只有有效的部分会被显示（没有黑色的部分）,如果设置为1,那么就会显示整个图像.
    设置为0~1之间的某个值,其效果也居于两者之间.
    newImageSize-校正后的图像分辨率,默认为原分辨率大小.
    validPixROI1-可选的输出参数,Rect型数据.其内部的所有像素都有效
    validPixROI2-可选的输出参数,Rect型数据.其内部的所有像素都有效
    """
    if type(height) != "int" or type(width) != "int":
        height = int(height)
        width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion,
        (width, height), R, T.T, alpha=0.5)
    """
    initUndistortRectifyMap
    cameraMatrix-摄像机参数矩阵
    distCoeffs-畸变参数矩阵
    R- stereoCalibrate() 求得的R矩阵
    newCameraMatrix-矫正后的摄像机矩阵(可省略)
    Size-没有矫正图像的分辨率
    m1type-第一个输出映射的数据类型,可以为 CV_32FC1或CV_16SC2 
    map1-输出的第一个映射变换
    map2-输出的第二个映射变换
    """
    map1x, map1y = cv2.initUndistortRectifyMap(
        left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    """
    cv2.remap重映射,就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程
    
    """
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


#视差计算
def sgbm(imgL, imgR):
    #SGBM参数设置
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity = 1,
                                   numDisparities = 64,
                                   blockSize = blockSize,
                                   P1 = 8 * img_channels * blockSize * blockSize,
                                   P2 = 32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff = -1,
                                   preFilterCap = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 100,
                                   mode = cv2.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.) # 除以16得到真实视差图
    return disp


#计算三维坐标,并删除错误点
def threeD(disp, Q):
    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q) 

    points_3d = points_3d.reshape(points_3d.shape[0] * points_3d.shape[1], 3)

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    #选择并删除错误的点
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    points_3d = np.delete(points_3d, remove_idx, 0)

    #计算目标点（这里我选择的是目标区域的中位数,可根据实际情况选取）
    if points_3d.any():
        x = np.median(points_3d[:, 0])
        y = np.median(points_3d[:, 1])
        z = np.median(points_3d[:, 2])
        targetPoint = [x, y, z]
    else:
        targetPoint = [0, 0, -1]#无法识别目标区域

    return targetPoint


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (
            2 * width, line_interval * (k + 1)), (0, 255, 0),
            thickness=2, lineType=cv2.LINE_AA)
 


imgL = cv2.imread("D:/cxn_project/Strain-gauges-recognition/cali_img/left/l6.bmp")
imgR = cv2.imread("D:/cxn_project/Strain-gauges-recognition/cali_img/right/r6.bmp")

height, width = imgL.shape[0:2]
# 读取相机内参和外参
config = stereoCameral()

map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
iml_rectified, imr_rectified = rectifyImage(imgL, imgR, map1x,
                                            map1y, map2x, map2y)

disp = sgbm(iml_rectified, imr_rectified)
plt.imshow(disp)
target_point = threeD(disp, Q) # 计算目标点的3D坐标（左相机坐标系下）
print(target_point)

