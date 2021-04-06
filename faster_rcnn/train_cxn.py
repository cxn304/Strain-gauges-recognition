'''
author:cxn rotate faster r-cnn train
本程序有两项loss,一项为判断每个anchors是否为正anchor的损失,另一项为正anchors要经过
5个参数的变换成为truth的5个回归项损失.truth是通过计算得到的anchor对应的5个回归变量
'''
import cv2
import random
import tensorflow as tf
import numpy as np
from utils import *
import pathlib
import multiprocessing as mp


'''
全局变量们,常量
'''
core_num = mp.cpu_count()
pos_thresh = 0.2
neg_thresh = 0.1
grid_width = grid_height = 16
image_height, image_width = 1024, 1280
cheng_length = len(chengG)
grid_w_num = int(image_width/grid_width)
grid_h_num = int(image_height/grid_height)


def encode_multiprocessing(gt_boxes):
    start1,end1 = 1,int(grid_h_num/4)   # 只分配h方向作为多核切割
    start2,end2 = int(grid_h_num/4),int(grid_h_num/2)
    start3,end3 = int(grid_h_num/2),int(grid_h_num/4*3)
    start4,end4 = int(grid_h_num/4*3),grid_h_num-1
    target_scores_q = mp.Queue()
    target_bboxes_q = mp.Queue()
    target_masks_q = mp.Queue()
    p1 = mp.Process(target=encode_label, args=(gt_boxes,start1,end1,target_scores_q,
                 target_bboxes_q,target_masks_q))
    p1.start()
    p2 = mp.Process(target=encode_label, args=(gt_boxes,start2,end2,target_scores_q,
                 target_bboxes_q,target_masks_q))
    p2.start()
    p3 = mp.Process(target=encode_label, args=(gt_boxes,start3,end3,target_scores_q,
                 target_bboxes_q,target_masks_q))
    p3.start()
    p4 = mp.Process(target=encode_label, args=(gt_boxes,start4,end4,target_scores_q,
                 target_bboxes_q,target_masks_q))
    p4.start()
    target_scores = [target_scores_q.get() for j in range(4)]
    target_bboxes = [target_bboxes_q.get() for j in range(4)]
    target_masks = [target_masks_q.get() for j in range(4)]
    '''
    取完才能join否则死锁
    '''
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    return sum(target_scores), sum(target_bboxes), sum(target_masks)


def encode_label(gt_boxes,start,end,target_scores_q,
                 target_bboxes_q,target_masks_q):
    '''
    输出target_scores, target_bboxes, target_masks, 即训练集真值数据
    '''
    target_scores = np.zeros(
        shape=[grid_h_num, 
               grid_w_num, cheng_length, 2]) # 0: background, 1: foreground, 
    target_bboxes = np.zeros(
        shape=[grid_h_num, 
               grid_w_num, cheng_length, 5]) # t_x,t_y,t_w,t_h,θ
    target_masks  = np.zeros(
        shape=[grid_h_num,
               grid_w_num, cheng_length]) # negative_samples: -1, positive_samples: 1
    for i in range(start,end): # y: height
        for j in range(1,grid_w_num-1): # x: width
            for k,wh in enumerate(chengG):
                center_x = j * grid_width + grid_width * 0.5
                center_y = i * grid_height + grid_height * 0.5
                xy_center = (center_x,center_y)
                xy_wh = tuple(wh[:2])
                xy_theta = wh[-1]   # 此时是弧度角,需要转换为角度
                xy_theta = xy_theta*180/3.14159
                minRect = (xy_center,xy_wh,xy_theta)
                rectCnt = np.int64(cv2.boxPoints(minRect))  # cv2的求法
                rectCnt = rectCnt.reshape(-1)   # 每个预设框的坐标
                boxes1=np.concatenate((rectCnt,[center_x,center_y],wh),axis=0)
                ious = compute_iou_rotate(boxes1, gt_boxes)
                positive_masks = ious > pos_thresh
                negative_masks = ious < neg_thresh
                
                if np.any(positive_masks):
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
    target_scores_q.put(target_scores)   
    target_bboxes_q.put(target_bboxes)
    target_masks_q.put(target_masks)
    # return target_scores, target_bboxes, target_masks


def process_image_label(image_path,rect_label_path,corner_label_path):
    """
    读取图片及其信息
    """
    raw_image = cv2.imread(image_path)  # 只读一幅图,三通道的RGB图
    gt_boxes = extract_r_boxes(rect_label_path,corner_label_path) # 只读那幅图的信息
    target = encode_multiprocessing(gt_boxes) # target_scores, target_bboxes, target_masks
    return raw_image/255., target  # 图片都要归一化


def create_image_label_path_generator(image_path,label_path):
    """
    建立迭代器
    """
    img_p = pathlib.Path(image_path)
    label_p = pathlib.Path(label_path)
    all_image_paths = [str(path) for path in img_p.iterdir()]
    all_label_paths = [str(path) for path in label_p.iterdir()]
    all_label_paths.sort()
    all_rect_label = [path for path in all_label_paths if 'txt' in path]
    all_corner_label = [path for path in all_label_paths if 'npz' in path]
    image_label_path = [[
    all_image_paths[i],all_rect_label[i],all_corner_label[i]] for i in range(
            len(all_image_paths))]  # 每组数据是一个整体
    while True:
        random.shuffle(image_label_path)
        for i in range(len(all_image_paths)):
            yield image_label_path[i]


def DataGenerator(image_path,label_path, batch_size):
    """
    generate image and mask at the same time
    一次返回batch_size数量的数据组
    """
    image_label_path_generator = create_image_label_path_generator(
        image_path,label_path) # 获取所有文件的路径
    while True:
        images = np.zeros(shape=[batch_size, image_height, image_width, 3], 
                          dtype=np.float)
        target_scores = np.zeros(
            shape=[batch_size, grid_h_num, 
                   grid_w_num, cheng_length, 2], dtype=np.float)
        target_bboxes = np.zeros(
            shape=[batch_size, grid_h_num, 
                   grid_w_num, cheng_length, 5],dtype=np.float)
        target_masks  = np.zeros(
            shape=[batch_size, grid_h_num,
                   grid_w_num, cheng_length], dtype=np.int)

        for i in range(batch_size):
            # next() 返回迭代器的下一个项目
            image_path, rect_label_path,corner_label_path = next(
                image_label_path_generator)
            image, target = process_image_label(image_path, rect_label_path,corner_label_path)
            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i]  = target[2]
        yield images, target_scores, target_bboxes, target_masks


def compute_loss(target_scores, target_bboxes, 
                 target_masks, pred_scores, pred_bboxes):
    """
    target_scores shape: [1, 64, 80, 12, 2],pred_scores shape: [1,64,80,12,2]
    target_bboxes shape: [1, 64, 80, 12, 5],pred_bboxes shape: [1,64,80,12,5]
    target_masks  shape: [1, 64, 80, 12]
    target_bboxes: t_x, t_y, t_w, t_h, t_theta两个平移量和两个尺度因子
    """
    score_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_scores, logits=pred_scores)
    """
    softmax_cross_entropy_with_logits计算得分的交叉熵损失,默认axis=-1,
    这里有[1, 64, 80, 12]个样本,每个样本有2个数,最后输出每个样本的交叉熵损失,
    所以输出结构为[1, 64, 80, 12]
    """
    foreground_background_mask = (np.abs(target_masks) == 1).astype(np.int)
    score_loss = tf.reduce_sum(
        score_loss * foreground_background_mask, axis=[1,2,3]) / np.sum(
            foreground_background_mask) # reduce_sum,axis=[1,2,3]表示对三个轴的损失求和
    score_loss = tf.reduce_mean(score_loss)

    boxes_loss = tf.abs(target_bboxes - pred_bboxes)    # tf.cast是数据类型转换
    boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(
        boxes_loss<1, tf.float32) + (boxes_loss - 0.5) * tf.cast(
            boxes_loss >=1, tf.float32) # boxes_loss<1和boxes_loss>=1是logic矩阵
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1) # 对最后一轴求和,得[1,64,80,12]
    foreground_mask = (target_masks > 0).astype(np.float32) # 找出物体位置
    boxes_loss = tf.reduce_sum(
        boxes_loss * foreground_mask, axis=[1,2,3]) / np.sum(foreground_mask)
    boxes_loss = tf.reduce_mean(boxes_loss)
    # 返回的是scoreloss和boxloss
    return score_loss, boxes_loss


if __name__ == '__main__':
    EPOCHS = 10 # 所有数据训练10遍
    STEPS = 32  # 30*2=60个数据
    batch_size = 8
    lambda_scale = 1.
    image_path,label_path = generate_train_img,generate_train_label
    TrainSet = DataGenerator(image_path,label_path, batch_size)  
    # 取出batch_size大小的iteration数据
    
    model = RPNplus()   # 类的实例化,model输出的是pred_scores, pred_bboxes
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    writer = tf.summary.create_file_writer("./log")
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64) 
    # tf的变量,不需要训练
    
    for epoch in range(EPOCHS):
        for step in range(STEPS):   # 遍历一次所有文件
            global_steps.assign_add(1)  # 每运行一次计数加1
            image_data, target_scores, target_bboxes, target_masks = next(TrainSet)
            image_data = image_data[...,0]  # 变为1通道的灰度图
            image_data = np.expand_dims(image_data, -1) # 还是用灰度图训练
            # 用迭代器每次取出batch_size数量的训练数据
            with tf.GradientTape() as tape:
                pred_scores, pred_bboxes = model(image_data) # Forward pass
                score_loss, boxes_loss = compute_loss(
                    target_scores, target_bboxes, target_masks,
                    pred_scores, pred_bboxes)   # 这一步拿模型输出的pred计算loss
                total_loss = score_loss + lambda_scale * boxes_loss
                # 用loss对trainable_variables即所有可训练的变量求导
                gradients = tape.gradient(total_loss, model.trainable_variables)
                # apply_gradients将计算得到的gradient和variables作为输入参数进行更新
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print("=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f" %(epoch+1, step+1,
                                                            total_loss.numpy(),
                                                            score_loss.numpy(), 
                                                            boxes_loss.numpy()))
            # writing summary data
            with writer.as_default():
                tf.summary.scalar("total_loss", total_loss, step=global_steps)
                tf.summary.scalar("score_loss", score_loss, step=global_steps)
                tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
            writer.flush()
            if global_steps % 10 == 0:  # 每过10步储存一次
                model.save_weights("RPN.h5")
        model.save_weights("RPN.h5")
