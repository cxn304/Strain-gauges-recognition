import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import math
import matplotlib.pyplot as plt
from utils import compute_iou_rotate, plot_rboxes_on_image, chengG, extract_r_boxes, compute_regression_rotate, decode_output_rotate

    

pos_thresh = 0.2
neg_thresh = 0.1
iou_thresh = 0.5
grid_width = 16  # 网格的长宽都是16,因为从原始图片到 feature map 经历了16倍的缩放
grid_height = 16
corner_num = len(chengG)
image_path = 'D:/cxn_project/Strain-gauges-recognition/colab_files/lb2.bmp'
rectlabel_path = 'D:/cxn_project/Strain-gauges-recognition/colab_files/lb2_rect.txt'
cornerlabel_path = 'D:/cxn_project/Strain-gauges-recognition/colab_files/lb2_corner.npz'
gt_boxes = extract_r_boxes(rectlabel_path,cornerlabel_path)
gt_boxes_corner = gt_boxes[:,:8]    # 画图用boxes_corner
image_width,image_height = 1280,1024

raw_image = cv2.imread(image_path)  # 将图片读取出来 (高，宽，通道数)
image_with_gt_boxes = np.copy(raw_image)  # 复制原始图片
canvas = plot_rboxes_on_image(image_with_gt_boxes, gt_boxes_corner)
plt.imshow(canvas)
plt.figure()
target_scores = np.zeros(shape=[64, 80, corner_num, 2]) # 0:background,1:foreground,
target_bboxes = np.zeros(shape=[64, 80, corner_num, 5]) # t_x, t_y, t_w, t_h,theta
target_masks  = np.zeros(shape=[64, 80, corner_num])
rects = np.zeros([1, 64, 80, corner_num, 4,2])
encoded_image = np.copy(raw_image)
grid_x, grid_y = tf.range(80, dtype=tf.int32), tf.range(64, dtype=tf.int32)
grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
grid_xy = tf.stack([grid_x, grid_y], axis=-1)
center_xy = grid_xy * 16 + 8   # 每个小框的中心位置
center_xy = tf.cast(center_xy, tf.float32)
huatubox = np.zeros([1,8])  # 测试chengG是否正确
for i in range(1,64-1):
    for j in range(1,80-1):
        for k,wh in enumerate(chengG):
            xy_center = tuple(center_xy[i,j,0,:])
            xy_wh = tuple(wh[:2])
            xy_theta = wh[-1]   # 此时是弧度角,需要转换为角度
            xy_theta = xy_theta*180/3.14159
            minRect = (xy_center,xy_wh,xy_theta)
            rectCnt = np.int64(cv2.boxPoints(minRect))  # cv2的求法
            rectCnt = rectCnt.reshape(-1)   # 每个预设框的坐标
            boxes1 = np.concatenate((rectCnt,center_xy[i,j,0,:],wh),axis=0)
            '''
            计算当前boxe和所有真实boxes的iou
            '''
            '''
            if i == 50 and j == 50:
                huatubox = np.concatenate(
                    (huatubox,boxes1[:8].reshape(1,8)),axis=0)
            '''
                
            ious = compute_iou_rotate(boxes1, gt_boxes)
            positive_masks = ious > pos_thresh
            negative_masks = ious < neg_thresh
            corners = boxes1[:8].reshape((4,2))
            
            if np.any(positive_masks):
                canvas = plot_rboxes_on_image(encoded_image,boxes1[:8].reshape(1,8), 
                                     thickness=1)
                plt.figure()
                plt.imshow(canvas)
                print("=> Encoding positive sample: %d, %d, %d" %(i, j, k))
                cv2.circle(encoded_image, center=(int(np.mean(corners[:,0])), 
                                                  int(np.mean(corners[:,1]))),
                                                  radius=1,
                                                  color=[255,0,0], 
                                                  thickness=4) # 正预测框的中心点用红圆表示

                target_scores[i, j, k, 1] = 1.  # 表示检测到物体
                target_masks[i, j, k] = 1 # labeled as a positive sample
                # find out which ground-truth box matches this anchor
                max_iou_idx = np.argmax(ious)   # 找到是哪个真实框
                selected_gt_boxes = gt_boxes[max_iou_idx]
                target_bboxes[i, j, k] = compute_regression_rotate(
                    selected_gt_boxes, boxes1)
                
            if np.all(negative_masks): # negative_masks数量为一幅图有几个真实框
                target_scores[i, j, k, 0] = 1.  # 表示是背景
                target_masks[i, j, k] = -1 # labeled as a negative sample
                
                cv2.circle(encoded_image, center=(int(np.mean(corners[:,0])), 
                                                  int(np.mean(corners[:,1]))),
                                radius=1, color=[0,0,0], thickness=4)
                
Image.fromarray(encoded_image).show()
'''                
plt.figure()
canvas = plot_rboxes_on_image(encoded_image,huatubox[1:], 
                                     thickness=1)  
plt.imshow(canvas)        
'''


# 现在的问题是只能找到其中的一个片子,然后没有继续往下计算