'https://blog.csdn.net/qq_36758914/article/details/105886811#3plot_boxes_on_image_53'
import numpy as np
import os
import xml.etree.ElementTree as ElementTree
import tensorflow as tf
import cv2

"""
wandhG 中包含着 9 个预测框的宽度和长度
"""
wandhG = np.array([[74., 149.],
                   [34., 149.],
                   [86.,  74.],
                   [109., 132.],
                   [172., 183.],
                   [103., 229.],
                   [149.,  91.],
                   [51., 132.],
                   [57., 200.]], dtype=np.float32)
wandhG = np.floor(wandhG*0.7, dtype=np.float32)

# 从注解文件中提取边框值
def extract_boxes(filename):
    # 加载并解析文件
    tree = ElementTree.parse(filename)
    # 获取文档根元素
    root = tree.getroot()
    # 提取出每个 bounding box 元素
    boxes = np.zeros([len(root.findall('.//bndbox')), 4])
    n=0
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = np.array([xmin, ymin, xmax, ymax])
        boxes[n,:] = coors
        n=n+1
    # 提取出图像尺寸
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes


def plot_boxes_on_image(show_image_with_boxes, boxes, color=[0, 0, 255], thickness=2):
    """
    plot_boxes_on_image() 函数的输入有两个,分别是:需要被画上检测框的原始图片以及
    检测框的左上角和右下角的坐标,
    其输出为被画上检测框的图片,直接改变show_image_with_boxes数组,外面是可以直接变的
    """
    for box in boxes:
        cv2.rectangle(show_image_with_boxes,
                      pt1=(int(box[0]), int(box[1])),
                      pt2=(int(box[2]), int(box[3])), color=color, 
                      thickness=thickness)
    show_image_with_boxes = cv2.cvtColor(
        show_image_with_boxes, cv2.COLOR_BGR2RGB)
    return show_image_with_boxes


def compute_iou(boxes1, boxes2):
    """
    compute_iou() 函数用来计算 IOU 值,即真实检测框与预测检测框(
        当然也可以是任意两个检测框)的交集面积比上
    它们的并集面积,这个值越大,代表这个预测框与真实框的位置越接近
    如果说得到的 IOU 值大于设置的正阈值,那么我们称这个预测框为正预测框(
        positive anchor)其中包含着检测目标;
    如果说得到的 IOU 值小于于设置的负阈值,那么我们称这个预测框为负预测框(
        )negative anchor),其中包含着背景
    """
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2] )
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = np.maximum(right_down - left_up, 0.0)  # 交集的宽和高
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]  # 交集的面积

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # anchor 的面积
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # ground truth boxes 的面积

    union_area = boxes1_area + boxes2_area - inter_area  # 并集的面积
    ious = inter_area / union_area
    return ious


def compute_regression(box1, box2):
    """
    正预测框与真实框之间的平移量和尺度因子间转换
    """
    target_reg = np.zeros(shape=[4, ])
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    target_reg[0] = (box1[0] - box2[0]) / w2  # 计算左上角的点位移差值
    target_reg[1] = (box1[1] - box2[1]) / h2
    target_reg[2] = np.log(w1 / w2)
    target_reg[3] = np.log(h1 / h2)

    return target_reg


def decode_output(pred_bboxes, pred_scores, score_thresh=0.5):
    """
    将一张图片上的 64*80*9 个预测框的平移量与尺度因子以及每个框的得分输入,
    得到每个正预测框对应的回归框(其实所有表示同一个检测目标的回归框都是近似重合的).
    pred_bboxes:它的形状为 [1, 64, 80, 9, 4],表示一共 64*80*9 个预测框,
    每个预测框都包含着两个平移量和两个尺度因子.
    pred_scores：它的形状为 [1, 64, 80, 9, 2]，表示在 64*80*9 个预测框中,
    [1, i, j, k, 0] 表示第 i 行第 j 列中的第 k 个预测框中包含的是背景的概率;
    [1, i, j, k, 1] 表示第 i 行第 j 列中的第 k 个预测框中包含的是检测物体的概率
    """
    grid_x, grid_y = tf.range(80, dtype=tf.int32), tf.range(64, dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
    grid_xy = tf.stack([grid_x, grid_y], axis=-1)
    center_xy = grid_xy * 16 + 8
    center_xy = tf.cast(center_xy, tf.float32)
    anchor_xymin = center_xy - 0.5 * wandhG

    xy_min = pred_bboxes[..., 0:2] * wandhG[:, 0:2] + anchor_xymin 
    # 平移量,9个预测框尺寸,9个预测框左上角坐标
    
    xy_max = tf.exp(pred_bboxes[..., 2:4]) * wandhG[:, 0:2] + xy_min 
    # 之前是log,现在是exp,正预测框变换回真实框

    pred_bboxes = tf.concat([xy_min, xy_max], axis=-1)  # 在倒数第一个维度拼接
    pred_scores = pred_scores[..., 1]   # 是否是物体的概率
    score_mask = pred_scores > score_thresh
    pred_bboxes = tf.reshape(pred_bboxes[score_mask], shape=[-1, 4]).numpy()
    # 运行前pred_bboxes为[1, 64, 80, 9, 4],score_mask为[1, 64, 80, 9]的bool,
    # 所以pred_bboxes[score_mask]为1列数据,需要reshape成4列若干行的数据,表示
    # ymin,xmin,ymax,xmax
    
    pred_scores = tf.reshape(pred_scores[score_mask], shape=[-1, ]).numpy()
    """
	经过 decode_output 函数的输出为:
	pred_score：其形状为 [-1, ]，表示每个检测框中的内容是检测物的概率
	pred_bboxes：其形状为 [-1, 4]，表示每个检测框的左上角和右下角的坐标
    但是经过t_x, t_y, t_w, t_h两个平移量和两个尺度因子的修正后,同一个物体的
    正预测框基本重合
	"""
    return pred_scores, pred_bboxes


def nms(pred_boxes, pred_score, iou_thresh):
    """
    pred_boxes shape: [-1, 4]
    pred_score shape: [-1,]
    nms()函数的作用是从选出的正预测框中进一步选出最好的n个预测框,其中,n指图片中检测物的个数.其流程为:
    取出所有预测框中得分最高的一个,并将这个预测框跟其他的预测框进行 IOU 计算;
    将IOU值大于0.1的预测框视为与刚取出的得分最高的预测框表示了同一个检测物,故去掉;
    重复以上操作,直到所有其他的预测框都被去掉为止
    """
    selected_boxes = []
    while len(pred_boxes) > 0:
        max_idx = np.argmax(pred_score)
        selected_box = pred_boxes[max_idx]
        selected_boxes.append(selected_box)
        pred_boxes = np.concatenate(
            [pred_boxes[:max_idx], pred_boxes[max_idx+1:]])
        pred_score = np.concatenate(
            [pred_score[:max_idx], pred_score[max_idx+1:]])
        ious = compute_iou(selected_box, pred_boxes)
        iou_mask = ious <= 0.1  # 把IOU小于0.1的保留,即这个box外其他的物体
        pred_boxes = pred_boxes[iou_mask]
        pred_score = pred_score[iou_mask]

    selected_boxes = np.array(selected_boxes)
    return selected_boxes
