'''
author: cxn
version: 0.1.0
图像拼接
'''

import numpy as np
import argparse
import cv2, os
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