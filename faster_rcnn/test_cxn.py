import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from rpn_cxn import RPNplus
from utils import decode_output_rotate, plot_rboxes_on_image, nms_cxn


"""
读取图片;
将图片输入训练好的 RPN 网络并得到每个预测框的得分和训练变量;
将得到的预测框的得分输入 softmax 层,得到每个预测框中的内容是背景或检测物的概率;
将一张图片上的 raw*col*12个预测框的平移量与尺度因子以及上一步中得到的概率输入,
得到每个正预测框对应的回归框（其实所有表示同一个检测目标的回归框都是重合的）;
执行 nms_cxn() 函数,取出最优的 n 个预测框;
将预测框画在图片上,并保存.
"""
if __name__ == '__main__':
    test_image_path ="./test_img"
    test_iamges = os.listdir(test_image_path)
    prediction_result_path = "./prediction"
    if not os.path.exists(prediction_result_path): os.mkdir(prediction_result_path)
    
    model = RPNplus()
    model.build([None, 1024, 1280, 1])  # 用一个build语句就好
    model.load_weights("./RPN.h5")  # 初始化模型后就可以
    
    for idx,imgs in enumerate(test_iamges):    # 文件名
        if idx == 10:break
        image_path = os.path.join(test_image_path, imgs)
        raw_image = cv2.imread(image_path)
        raw_image = raw_image[...,0]
        image_data = np.expand_dims(raw_image / 255., 0)
        image_data = np.expand_dims(image_data, -1)
        pred_scores, pred_bboxes = model(image_data)
        # Softmax简单的说就是把一个N*1的向量归一化为(0，1)之间的值
        pred_scores = tf.nn.softmax(pred_scores, axis=-1) 
        pred_bboxes,pred_scores = decode_output_rotate(pred_bboxes, pred_scores)
        pred_bboxes = nms_cxn(pred_bboxes, pred_scores, 0.5)
        plot_rboxes_on_image(raw_image, pred_bboxes[:,:8])
        save_path = os.path.join(prediction_result_path, imgs)
        print("=> saving prediction results into %s" %save_path)
        Image.fromarray(raw_image).save(save_path)
