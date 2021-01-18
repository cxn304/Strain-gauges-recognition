'''
author: cxn
version: 0.1.0
cnn read file and label, for mask r-cnn
'''

import numpy as np 
import os
import xml.etree.ElementTree as ElementTree
from mrcnn.utils import Dataset
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import tensorflow as tf


# 用于定义和加载袋鼠数据集的类
class gaugesDataset(Dataset):
	# 加载数据集定义
	def load_dataset(self, dataset_dir, is_train=True):
		# 定义一个类
		self.add_class("dataset", 1, "gauges")
		# 定义数据所在位置
		images_dir = dataset_dir + '/train_image/'
		annotations_dir = dataset_dir + '/gauges_labels/'
		# 定位到所有图像
		for filename in os.listdir(images_dir):
			# 提取图像名称
			image_id = filename[:-4] # 后缀是.bmp这个长度的才可以

            # 如果我们正在建立的是测试/验证集，略过 15 序号之前的所有图像
# 			if not is_train and int(image_id) < 15:
# 				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# 添加到数据集
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)


	# 从注解文件中提取边框值
	def extract_boxes(self, filename):
		# 加载并解析文件
		tree = ElementTree.parse(filename)
		# 获取文档根元素
		root = tree.getroot()
		# 提取出每个 bounding box 元素
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# 提取出图像尺寸
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height


	# 加载图像掩膜
	def load_mask(self, image_id):
		# 获取图像详细信息
		info = self.image_info[image_id]
		# 定义盒文件位置
		path = info['annotation']
		# 加载 XML
		boxes, w, h = self.extract_boxes(path)
		# 为所有掩膜创建一个数组，每个数组都位于不同的通道
		masks = np.zeros([h, w, len(boxes)], dtype='uint8')
		# 创建掩膜
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('gauges'))
		return masks, np.asarray(class_ids, dtype='int32')


	# 加载图像引用
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


# 定义模型配置
class gaugesConfig(Config):
	# 定义配置名
	NAME = "gauge_cfg"
	# 类的数量（背景中的 + 袋鼠）
	NUM_CLASSES = 1 + 1
	# 每轮训练的迭代数量
	STEPS_PER_EPOCH = 20
    
    

# 训练集
train_set = gaugesDataset()
train_set.load_dataset('gauges', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# 准备测试/验证集
test_set = gaugesDataset()
test_set.load_dataset('gauges', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# 加载图像
image_id = 1
# 加载图像
image = train_set.load_image(image_id)
# 加载掩膜和类 id
mask, class_ids = train_set.load_mask(image_id)
# 从掩膜中提取边框, mcnn里面的
bbox = extract_bboxes(mask)
# 显示带有掩膜和边框的图像
display_instances(image, bbox, mask, class_ids, train_set.class_names)

# 准备配置信息
config = gaugesConfig()
config.display()
# 定义模型
model = MaskRCNN(mode='training', model_dir='./cnn_model', config=config)
# 训练权重（输出层，或者说‘头部’）
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, 
            epochs=5,layers="all")



        
        
        
        
        
        
        
        
        
        
        
        