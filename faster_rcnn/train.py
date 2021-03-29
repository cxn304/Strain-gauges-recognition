import os, glob
import cv2
import random
import tensorflow as tf
import numpy as np
from utils import compute_iou, extract_boxes, wandhG, compute_regression
from rpn import RPNplus


pos_thresh = 0.5
neg_thresh = 0.1
grid_width = grid_height = 16
image_height, image_width = 1024, 1280


def encode_label(gt_boxes):
    target_scores = np.zeros(shape=[64, 80, 9, 2]) # 0: background, 1: foreground, ,
    target_bboxes = np.zeros(shape=[64, 80, 9, 4]) # t_x, t_y, t_w, t_h
    target_masks  = np.zeros(shape=[64, 80, 9]) # negative_samples: -1, positive_samples: 1
    for i in range(64): # y: height
        for j in range(80): # x: width
            for k in range(9):
                center_x = j * grid_width + grid_width * 0.5
                center_y = i * grid_height + grid_height * 0.5
                xmin = center_x - wandhG[k][0] * 0.5
                ymin = center_y - wandhG[k][1] * 0.5
                xmax = center_x + wandhG[k][0] * 0.5
                ymax = center_y + wandhG[k][1] * 0.5
                # print(xmin, ymin, xmax, ymax)
                # ignore cross-boundary anchors
                if (xmin > -5) & (ymin > -5) & (
                        xmax < (image_width+5)) & (ymax < (image_height+5)):
                    anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                    anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                    # compute iou between this anchor and all ground-truth boxes in image.
                    '''
                    这里需要改,就是直接计算IOU
                    '''
                    ious = compute_iou(anchor_boxes, gt_boxes)
                    positive_masks = ious >= pos_thresh
                    negative_masks = ious <= neg_thresh

                    if np.any(positive_masks):
                        target_scores[i, j, k, 1] = 1.
                        target_masks[i, j, k] = 1 # labeled as a positive sample
                        # find out which ground-truth box matches this anchor
                        max_iou_idx = np.argmax(ious)
                        selected_gt_boxes = gt_boxes[max_iou_idx]
                        target_bboxes[i, j, k] = compute_regression(
                            selected_gt_boxes, anchor_boxes[0])

                    if np.all(negative_masks):
                        target_scores[i, j, k, 0] = 1.
                        target_masks[i, j, k] = -1 # labeled as a negative sample
                        # target_masks == 0表示既不是背景也不是检测物  
    return target_scores, target_bboxes, target_masks


def process_image_label(image_path, label_path):
    """
    读取图片及其信息
    """
    raw_image = cv2.imread(image_path)  # 只读一幅图,三通道的RGB图
    gt_boxes = extract_boxes(label_path) # 只读那幅图的信息
    target = encode_label(gt_boxes) # target_scores, target_bboxes, target_masks
    return raw_image/255., target  # 图片都要归一化


def create_image_label_path_generator(synthetic_dataset_path):
    """
    建立迭代器
    """
    image_num = 60
    photo_path = synthetic_dataset_path + '\\train_image\\'
    label_path = synthetic_dataset_path + '\\gauges_labels\\'
    image_label_paths = [glob.glob(photo_path + '*.bmp'),
                         glob.glob(label_path + '*.xml')]
    image_label_path = [[image_label_paths[0][i],image_label_paths[1][i]] for i in range(len(image_label_paths[0]))]
    while True:
        random.shuffle(image_label_path)
        for i in range(image_num):
            yield image_label_path[i]


def DataGenerator(synthetic_dataset_path, batch_size):
    """
    generate image and mask at the same time
    synthetic_dataset_path = ../gauges
    一次返回batch_size数量的数据组
    """
    image_label_path_generator = create_image_label_path_generator(
        synthetic_dataset_path) # 获取所有文件的路径
    while True:
        images = np.zeros(shape=[batch_size, image_height, image_width, 3], 
                          dtype=np.float)
        target_scores = np.zeros(shape=[batch_size, 64, 80, 9, 2], 
                                 dtype=np.float)
        target_bboxes = np.zeros(shape=[batch_size, 64, 80, 9, 4],
                                 dtype=np.float)
        target_masks  = np.zeros(shape=[batch_size, 64, 80, 9], 
                                 dtype=np.int)

        for i in range(batch_size):
            # next() 返回迭代器的下一个项目
            image_path, label_path = next(image_label_path_generator)
            image, target = process_image_label(image_path, label_path)
            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i]  = target[2]
        yield images, target_scores, target_bboxes, target_masks


def compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes):
    """
    target_scores shape: [1, 64, 80, 9, 2],  pred_scores shape: [1,64,80,9,2]
    target_bboxes shape: [1, 64, 80, 9, 4],  pred_bboxes shape: [1,64,80,9,4]
    target_masks  shape: [1, 64, 80, 9]
    target_bboxes: t_x, t_y, t_w, t_h两个平移量和两个尺度因子
    """
    score_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_scores, logits=pred_scores)
    """
    softmax_cross_entropy_with_logits计算得分的交叉熵损失,默认axis=-1,
    这里有[1, 64, 80, 9]个样本,每个样本有2个数,最后输出每个样本的交叉熵损失,
    所以输出结构为[1, 64, 80, 9]
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
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1) # 对最后一轴求和,得[1, 64, 80, 9]
    foreground_mask = (target_masks > 0).astype(np.float32) # 找出物体位置
    boxes_loss = tf.reduce_sum(
        boxes_loss * foreground_mask, axis=[1,2,3]) / np.sum(foreground_mask)
    boxes_loss = tf.reduce_mean(boxes_loss)
    # 返回的是一个平均loss,即一个float数
    return score_loss, boxes_loss



EPOCHS = 10 # 所有数据训练10遍
STEPS = 35  # 30*2=60个数据
batch_size = 2
lambda_scale = 1.
synthetic_dataset_path="..\\gauges"
TrainSet = DataGenerator(synthetic_dataset_path, batch_size)  
# 取出batch_size大小的iteration数据

model = RPNplus()   # 类的实例化,model输出的是pred_scores, pred_bboxes
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
writer = tf.summary.create_file_writer("./log")
global_steps = tf.Variable(0, trainable=False, dtype=tf.int64) # tf的变量,不需要训练

for epoch in range(EPOCHS):
    for step in range(STEPS):   # 遍历一次所有文件
        global_steps.assign_add(1)  # 每运行一次计数加1
        image_data, target_scores, target_bboxes, target_masks = next(TrainSet)
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
    model.save_weights("RPN.h5")
