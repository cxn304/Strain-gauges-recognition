'my code using rotated boxes'
import numpy as np
import random,pathlib, os
import xml.etree.ElementTree as ElementTree
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

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

chengG = np.array([[70., 140., 0],
                   [70., 140., 3.14/6],
                   [70., 140., -3.14/6],
                   [140.,  70., 0],
                   [140., 70., 3.14/6],
                   [140., 70., -3.14/6],
                   [48., 87., 0],
                   [48., 87., 3.14/6],
                   [48., 87., -3.14/6],
                   [87.,  48., 0],
                   [87., 48., 3.14/6],
                  [87., 48., -3.14/6]],
                  dtype=np.float32)


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
    return boxes,width,height


# 从注解文件中提取真实框的数据
def extract_r_boxes(filename_rect,filename_corner):
    '''
    旋转值提取函数,rect(five_data).txt,corner(four_corner).npz
    '''
    corner = np.load(filename_corner,allow_pickle=True)
    gauge_recs = corner.files # gauge rec list name
    boxes = np.zeros([len(gauge_recs), 13])
    with open(filename_rect, 'r') as f:
        data = f.read() # 读取txt的data
    data = data.split('\n')[:-1]
    for n,value in enumerate(data):
        xyh = value.split(';')
        x,y = xyh[0].strip('()').split(',')
        x = int(float(x))
        y = int(float(y))
        w,h = xyh[1].strip('()').split(',')
        w = int(float(w)) # 宽
        h = int(float(h)) # 高
        theta = xyh[2]
        theta= int(float(theta)) # 度
        rects = np.array([x,y,w,h,theta])
        boxes[n,8:] = rects
    
    for n,c in enumerate(gauge_recs):
        boxes[n,0:2] = corner[c][0]
        boxes[n,2:4] = corner[c][1]
        boxes[n,4:6] = corner[c][2]
        boxes[n,6:8] = corner[c][3]
    
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


def plot_rboxes_on_image(img, boxes, color=[0, 255, 0], thickness=2):
    """
    旋转的boxes的结构为外接矩形四个点坐标,boxes=[n,8] or [n,4,2]
    """
    if len(boxes.shape) == 2:
        h,l = boxes.shape
        boxes = tf.reshape(boxes,shape=[h,4,2]).numpy()
    boxes = boxes.astype(int)
    canvas = np.copy(img)
    for i in range(len(boxes)): # len(boxes)表示第一维度的长度
        cv2.drawContours(canvas, [boxes[i,:,:]], 0, color, 2)
    return canvas


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


def compute_iou_rotate(boxeso1,boxeso2):
    """
    分割法计算IOU
    boxes2为真实值
    boxes1,2是列数为13的
    """
    n = len(boxeso2)
    ious = np.zeros(n)
    w1 = boxeso1[10]  # w设为width
    h1 = boxeso1[11]  # h设为height
    for ii in range(n): 
        w2 = boxeso2[ii,10]  # w设为width
        h2 = boxeso2[ii,11]
        if (abs(boxeso1[8]-boxeso2[ii,8])<24 
            and abs(boxeso1[9]-boxeso2[ii,9])<24):
            a1box = []  # 一般式的a,有四个因为有四条边
            a2box = []  
            b1box = []
            b2box = []
            c1box = []
            c2box = []
            
            boxes1 = boxeso1[:8].reshape(4,2)
            boxes2 = boxeso2[ii,:8].reshape(4,2)
            # boxes2 = boxeso2[:,:8].reshape(n,4,2)
            boxes1 = np.concatenate((boxes1,boxes1[0,:].reshape((1,2))),axis=0)
            boxes2 = np.concatenate((boxes2,boxes2[0,:].reshape((1,2))),axis=0)
            # boxes2 = np.concatenate((boxes2,boxes2[:,0,:].reshape((n,1,2))),axis=1)
            '''
            两点式转为一般式的abc
            ''' 
            for i in range(4):  
                a1box.append(-(boxes1[i,1]-boxes1[i+1,1]))
                a2box.append(-(boxes2[i,1]-boxes2[i+1,1]))
                # a2box.append(-(boxes2[:,0,1]-boxes2[:,0+1,1]))
                b1box.append((boxes1[i,0]-boxes1[i+1,0]))
                b2box.append((boxes2[i,0]-boxes2[i+1,0]))
                # b2box.append((boxes2[:,i,0]-boxes2[:,i+1,0]))
                c1box.append(boxes1[i,1]*(
                    boxes1[i+1,0]-boxes1[i,0])-boxes1[i,0]*(
                        boxes1[i+1,1]-boxes1[i,1]))
                
                c2box.append(boxes2[i,1]*(
                    boxes2[i+1,0]-boxes2[i,0])-boxes2[i,0]*(
                        boxes2[i+1,1]-boxes2[i,1]))
                '''
                c2box.append(boxes2[:,i,1]*(
                    boxes2[:,i+1,0]-boxes2[:,i,0])-boxes2[:,i,0]*(
                        boxes2[:,i+1,1]-boxes2[:,i,1]))
                '''
            tmp_point = []
            for i in range(4):  # 找到两个矩形间的所有交点
                for j in range(4):
                    w = a1box[i]*b2box[j] - b1box[i]*a2box[j] 
                    # w = p1.a*p2.b - p1.b*p2.a
                    if w*w < 0.01: w = 0.1
                    # 找到交点
                    intersectionx = (b1box[i]*c2box[j]-c1box[i]*b2box[j])/w  
                    # (p1.b*p2.c - p1.c*p2.b) / w
                    intersectiony = (c1box[i]*a2box[j]-a1box[i]*c2box[j])/w  
                    # (p1.c*p2.a - p1.a*p2.c) / w
                    '''
                    若交点在两条线段之间
                    '''
                    if intersectionx<=max((boxes1[i,0],boxes1[i+1,0])) and intersectionx>=min((boxes1[i,0],boxes1[i+1,0])) and intersectiony<=max((boxes1[i,1],boxes1[i+1,1])) and intersectiony>=min((boxes1[i,1],boxes1[i+1,1])) and intersectionx<=max((boxes2[i,0],boxes2[i+1,0])) and intersectionx>=min((boxes2[i,0],boxes2[i+1,0])) and intersectiony<=max((boxes2[i,1],boxes2[i+1,1])) and intersectiony>=min((boxes2[i,1],boxes2[i+1,1])):
                       tmp_point.append((intersectionx,intersectiony)) 
                       
                    '''
                    若交点在两条线段的顶点处
                    '''
                    if (intersectionx==boxes1[i,0] and intersectiony==boxes1[i,1]) or (intersectionx==boxes1[i+1,0] and intersectiony==boxes1[i+1,1]) or (intersectionx==boxes2[i,0] and intersectiony==boxes2[i,1]) or (intersectionx==boxes2[i+1,0] and intersectiony==boxes2[i+1,1]):
                        tmp_point.append((intersectionx,intersectiony)) 
              
            if len(tmp_point)>2:
                nn = len(tmp_point) # 有几个交点
                tmp_point.append(tmp_point[0]) # 计算并集面积
                
                initarea = 0
                for i in range(nn):
                    initarea = initarea + tmp_point[i][0]*tmp_point[i+1][1]-tmp_point[i][1]*tmp_point[i+1][0]
                initarea = 0.5*abs(initarea)    # 并集面积
                ious[ii] = initarea/(w1*h1+w2*h2-initarea)   # 计算交并比
        
    return ious
    
    
def compute_regression_rotate(boxes1,boxes2):
    '''
    boxes2是chengG中的值
    这一步给出的值相当于y-truth
    '''
    target_reg = np.zeros(shape=[5, ])
    x1 = boxes1[8]
    y1 = boxes1[9]
    w1 = boxes1[10]
    h1 = boxes1[11]
    theta1 = boxes1[12]

    
    x2 = boxes2[8]
    y2 = boxes2[9]
    w2 = boxes2[10]
    h2 = boxes2[11]
    theta2 = boxes2[12]
    
    
    target_reg[0] = (x1 - x2)/(w2)
    target_reg[1] = (y1 - y2) / (h2)
    target_reg[2] = np.log(w1 / w2)
    target_reg[3] = np.log(h1 / h2)
    target_reg[4] = (theta1-theta2*180/3.14159)/60  
    # 这里的单位是度,解码时候要乘以60,
    
    return target_reg
    
    
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


def decode_output_rotate(pred_bboxes, pred_scores, score_thresh=0.5):
    """
    将一张图片上的 64*80*12个预测框的平移量与尺度因子以及每个框的得分输入,
    得到每个正预测框对应的回归框(其实所有表示同一个检测目标的回归框都是近似重合的).
    pred_bboxes:它的形状为 [1, 64, 80, 12, 5],表示一共 64*80*12 个预测框,
    每个预测框都包含着两个平移量和两个尺度因子还有一个旋转因子.
    pred_scores：它的形状为 [1, 64, 80, 12, 2]，表示在 64*80*12 个预测框中,
    [1, i, j, k, 0] 表示第 i 行第 j 列中的第 k 个预测框中包含的是背景的概率;
    [1, i, j, k, 1] 表示第 i 行第 j 列中的第 k 个预测框中包含的是检测物体的概率
    供画图使用,这里有问题,明天修改
    """
    used_chengG = []
    _,height,width,chengG_width,param = pred_bboxes.shape
    for i in range(height):
        for j in range(width):
            for k in range(chengG_width):
                if pred_scores[0,i,j,k,1] == 1:
                    used_chengG.append((pred_bboxes[0,i,j,k,:],i,j,k))  
                    # 把i,j,k储存下来的原因是弄清楚是chengG中的哪个框,以便做变换
    
    rects = np.zeros([chengG_width,4,2])
    grid_x, grid_y = tf.range(80, dtype=tf.int32), tf.range(64, dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
    grid_xy = tf.stack([grid_x, grid_y], axis=-1)
    center_xy = grid_xy * 16 + 8   # 每个小框的中心位置
    center_xy = tf.cast(center_xy, tf.float32)
    '''
    接下来要通过五个regression给出四个角点坐标
    '''
    for ck,final_rect in enumerate(used_chengG):
        trans,i,j,k = final_rect
        xy_center = tuple(center_xy[i,j,0,:]+chengG[k,:2]*trans[:2])
        xy_wh = tuple(chengG[k,:2]*tf.exp(trans[2:4]))
        xy_theta = trans[-1]*60+chengG[k,-1]*180/3.14159 
        # regression时候是-,现在是加
        minRect = (xy_center,xy_wh,xy_theta)
        rectCnt = np.int64(cv2.boxPoints(minRect))  # cv2的求法
        rects[ck,:,:] = rectCnt

    return rects


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


def generate_train_data(all_paths):
    '''
    生成训练集图片和标签还有mask,all_paths表示自己手工制作的原始训练集图片和标签
    的地址.
    '''
    n = len(all_paths)
    cn = random.choice(range(n))  # 选取哪个模板以加入mask和生成npy,npz
    filename_rect,filename_corner = all_paths[cn][1],all_paths[cn][2]
    filename_img,filename_mask = all_paths[cn][0],all_paths[cn][3]
    gt_boxes = extract_r_boxes(filename_rect,filename_corner)
    xbias = random.randint(-80,80)  # 这个偏差是由两幅图的位置决定的
    ybias = random.randint(-500,100)
    x_o = gt_boxes[:,8]   # 所有矩形坐标x
    y_o = gt_boxes[:,9]   # 所有矩形坐标y
    x_o = x_o + xbias   # 对几个矩形进行平移
    y_o = y_o + ybias
    mask = np.load(filename_mask) # 读取mask，为与图像大小一致的0-1矩阵
    gauge_img = plt.imread(filename_img)
    h,l = mask.shape # 行，列
    M = np.float32([[1,0,xbias],[0,1,ybias]]) # 平移矩阵
    new_mask = cv2.warpAffine(mask,M,(l,h)) # opencv按照矩阵进行mask平移
    new_gauge_img = cv2.warpAffine(gauge_img,M,(l,h)) # 原图进行平移
    boxes1 = np.zeros([len(x_o),13]) # 变换后的13长度的box
    for i in range(len(x_o)):
      xy_center = (x_o[i],y_o[i])
      xy_wh = tuple(gt_boxes[i][10:12])
      xy_theta = gt_boxes[i][-1]
      minRect = (xy_center,xy_wh,xy_theta)
      rectCnt = np.int64(cv2.boxPoints(minRect))  # cv2的求法
      rectCnt = rectCnt.reshape(-1)   # 每个预设框的坐标
      tmp = np.concatenate((rectCnt,[x_o[i],y_o[i]],gt_boxes[i,-3:]),axis=0)
      boxes1[i,:] = tmp # 这里没错误
    
    return new_gauge_img,new_mask,boxes1


def create_image_label_path(image_path,label_path):
    """
    建立提取训练图像及标签的数据:image,rect(five_data),corner(four_corner),mask
    """
    img_p = pathlib.Path(image_path)
    label_p = pathlib.Path(label_path)
    all_image_paths = [str(path) for path in img_p.iterdir()]
    all_label_paths = [str(path) for path in label_p.iterdir()]
    all_label_paths.sort()
    all_mask_paths = [path for path in all_label_paths if 'npy' in path]
    all_rect_label = [path for path in all_label_paths if 'txt' in path]
    all_corner_label = [path for path in all_label_paths if 'npz' in path]
    image_label_path = [[
      all_image_paths[i],all_rect_label[i],all_corner_label[i],
      all_mask_paths[i]] for i in range(len(all_image_paths))]
    return image_label_path


def add_gauge_to_img(filepath,image_path,label_path):
    """
    replace train image with gauges, it use generate_train_data output.
    filepath:COCO dataset path, image_path:gauges image path, label_path:
    gauges label path.
    """
    plt.figure(figsize=(10, 10))  # 设置图像大小
    allfile = os.listdir(filepath)
    allfile_num = len(allfile)
    if not os.path.exists('/content/train_img/'): #判断文件夹是否存在
        os.makedirs('/content/train_img/')
    if not os.path.exists('/content/train_labels/'): #判断文件夹是否存在
        os.makedirs('/content/train_labels/')
    for i,filename in enumerate(allfile):
        if i == 9:break
        saveimg = read_img_from_tf(filepath+filename)
        all_paths = create_image_label_path(image_path,label_path)
        new_gauge_img,new_mask,boxes1=generate_train_data(all_paths)  #################
        saveimg[new_mask==1] = new_gauge_img[new_mask==1]
        cv2.imwrite('/content/train_img/'+filename, saveimg)  # 储存加上应变片的训练图片
        # canvass = plot_gauges(saveimg)
        plt.subplot(3, 3, i + 1)
        plt.imshow(saveimg,cmap='gray')
        plt.title(str(i))
        plt.axis("off")
        save_contours_recs(filename,boxes1)
    

def save_contours_recs(filename,boxes1):
    '''
    boxes1为长度为[n,13]的numpy矩阵
    four_corner储存矩形四个点坐标以及....all_rect最小外接矩形的中心(x,y),(宽度,高度),旋转角度
    '''
    all_rect = []
    four_corner = []
    for i in range(len(boxes1)):
      four_corner.append(boxes1[i,:8].reshape(4,2))
      xy_center = (boxes1[i,8],boxes1[i,9])
      xy_wh = (boxes1[i,10],boxes1[i,11])
      xy_theta = boxes1[i,12]
      minAreaRect = (xy_center,xy_wh,xy_theta)
      all_rect.append(minAreaRect)

    this_name = filename.split('.')[0]
    np.savez('/content/train_labels/'+ this_name + '.npz', *four_corner) # 解包list
    rectname = open('/content/train_labels/'+ this_name + '.txt', 'w')
    # np.savez('./colab_files/'+ this_name + '_rect.npz', np.array(all_rect))
    for value in all_rect:
        for values in value:
            rectname.write(str(values))
            rectname.write(';')
        rectname.write('\n')
    rectname.close()


def read_img_from_tf(filename):
    """
    read file from tf and resize it to [1024, 1280]
    """
    img_raw = tf.io.read_file(filename) # 原始二进制数据
    img_tensor = tf.image.decode_image(img_raw) # 将它解码为图像tensor张量
    img_final = tf.image.resize(img_tensor, [1024, 1280])[:,:,0]
    saveimg = img_final.numpy() # 0-255
    return saveimg