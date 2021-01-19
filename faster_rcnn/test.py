import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from rpn import RPNplus
from utils import decode_output, plot_boxes_on_image, nms


"""
读取图片;
将图片输入训练好的 RPN 网络并得到每个预测框的得分和训练变量;
将得到的预测框的得分输入 softmax 层,得到每个预测框中的内容是背景或检测物的概率;
将一张图片上的 45*60*9 个预测框的平移量与尺度因子以及上一步中得到的概率输入,
得到每个正预测框对应的回归框（其实所有表示同一个检测目标的回归框都是重合的）;
执行 nms() 函数,取出最优的 n 个预测框;
将预测框画在图片上,并保存.
"""
synthetic_dataset_path ="./synthetic_dataset/synthetic_dataset"
prediction_result_path = "./prediction"
if not os.path.exists(prediction_result_path): os.mkdir(prediction_result_path)

model = RPNplus()
fake_data = np.ones(shape=[1, 720, 960, 3]).astype(np.float32)
model(fake_data) # initialize model to load weights
model.load_weights("./RPN.h5")

for idx in range(8000, 8200):
    image_path = os.path.join(synthetic_dataset_path, "image/%d.jpg" %(idx+1))
    raw_image = cv2.imread(image_path)
    image_data = np.expand_dims(raw_image / 255., 0)
    pred_scores, pred_bboxes = model(image_data)
    pred_scores = tf.nn.softmax(pred_scores, axis=-1)
    pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores, 0.9)
    pred_bboxes = nms(pred_bboxes, pred_scores, 0.5)
    plot_boxes_on_image(raw_image, pred_bboxes)
    save_path = os.path.join(prediction_result_path, str(idx)+".jpg")
    print("=> saving prediction results into %s" %save_path)
    Image.fromarray(raw_image).save(save_path)
