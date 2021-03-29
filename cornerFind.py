# -*- coding: UTF-8 -*-

'''
弱智的想法
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./capture_folder/save_img/l3.bmp')
img = img[:,:,0]

corners = cv2.goodFeaturesToTrack(img,90,0.2,7)
corners = corners[:,0,:]
plt.figure()
plt.imshow(img)
plt.scatter(corners[:,0],corners[:,1])