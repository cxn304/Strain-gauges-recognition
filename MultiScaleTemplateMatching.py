import numpy as np
import cv2, pyclipper
import matplotlib.pyplot as plt

#Using custom template for Set A (setA.png)
#Using custom template for Set B (t2.png for t2_x.jpg, t3.png for t3_x.jpg )
#Use your template for set B (t1_x.jpg images)


def Multi_scale_temp():
    #Multi-Scaling Template
    found = None
    for scale in np.linspace(0.1, 0.4,20)[::-1]:
        height, width = template.shape[:2]
        size = (int(width*scale), int(height*scale))
        resized = cv2.resize(template, size, interpolation=cv2.INTER_AREA)
        r = template.shape[1] / float(resized.shape[1])
        result = cv2.matchTemplate(blur_image, resized, cv2.TM_CCOEFF_NORMED)
    	#Finding most co-related point.
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    	
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, scale)
    (maxVal, maxLoc, r,scale) = found
    size = (int(width*scale), int(height*scale))
    
    # Draw a bounding box around the detected result
    # if greater than threshold.
    threshold = 0.8
    if(maxVal > threshold):
        print('Cursor Found!')
        cv2.circle(image, (int(maxLoc[0])+size[0]//2,
                           int(maxLoc[1])+size[1]//2), 3, (255, 0, 255), 3)
    else:
        print('Cursor not found')
        
    plt.imshow(image)
    
    

def find_all_gauges(blurs,imgss,LURD):
    """
    通过点击鼠标得到面积
    """
    LURD = np.array(LURD)
    mianji = abs((LURD[0]-LURD[1])[0]*(LURD[0]-LURD[1])[1]) # 应变片面积
    #mianji = 400
    blur = blurs.copy()
    imgs = imgss.copy()
    row,col = blurs.shape[:2]
    blurs = cv2.convertScaleAbs(blur,alpha=1.5,beta=0)
    # ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret,th = cv2.threshold(
    #     blur,blurs[LURD[2][1],LURD[2][0]],255,cv2.THRESH_BINARY)
    ret,th = cv2.threshold(blur,110,255,cv2.THRESH_BINARY)
    th[th==0]=1
    th[th==255]=0
    kernel = np.ones((3,3),np.uint8)    # 前面有腐蚀操作,这里要注意
    erosion = cv2.erode(th,kernel,iterations = 1)
    plt.figure()
    plt.imshow(erosion)
    find_gauges = np.zeros((row,col),np.uint8)
    num, labels = cv2.connectedComponents(erosion)
    all_labels = [(i,np.sum(labels==i)) for i in range(num)] # 所有连通域的面积
    # 只保留应变片
    x=[(i,np.sum(labels==i)) for i in range(num) 
       if np.sum(labels==i)>mianji*0.7 and np.sum(labels==i)<mianji*1.2]
    masks = create_mask_gauges(erosion,kernel,x,labels) # 所有应变片的掩膜
    np.save('mask_for_gauges1.npy',masks)
    for i in range(len(x)):
        find_gauges[labels==x[i][0]] = 1
    imgs[:,:,1] = blur
    # 寻找轮廓
    bimg, contours, hier = cv2.findContours(
        find_gauges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    std_error,all_rect,ratio_list = plot_rec_to_canvas(img,contours)
    
    return all_rect,ratio_list,std_error,all_labels
    
    
def create_mask_gauges(erosion,kernel,x,labels):  
    masks = erosion.copy()
    masks = masks*0
    for tt in x:
        masks[labels==tt[0]] = 1
    masks = cv2.dilate(masks,kernel,iterations = 1)
    
    return masks
    

def plot_rec_to_canvas(img,contours):
    canvas = np.copy(img)   # 声明画布拷贝自img
    ratio_list = [] # 每个图在图像中的长度和真实长度的比值,长和宽两个
    all_rect = []   # 每个矩形的信息
    four_corner = []
    for cidx,cnt in enumerate(contours):
        pco = pyclipper.PyclipperOffset()           # 多边形外扩函数
        pco.AddPath(cnt[:,0,:], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solu_contour = pco.Execute(1)   # 轮廓内缩或者外扩
        cnt_ = np.array(solu_contour)
        cnt__ = np.transpose(cnt_,(1,0,2))
        minAreaRect = cv2.minAreaRect(cnt__)  
        # 最小外接矩形的中心(x,y),(宽度,高度),旋转角度(逆时针)
        all_rect.append(minAreaRect)
        ratio_list.append(translate_w_h(minAreaRect))
        rectCnt = np.int64(cv2.boxPoints(minAreaRect))
        four_corner.append(rectCnt)
        cv2.drawContours(canvas, [rectCnt], 0, (0,255,0), 3)
        
    save_contours_recs(imagePath,all_rect,four_corner)
    plt.figure()
    plt.imshow(canvas)
    std_error = cal_gauges_merror(all_rect)
    return std_error,all_rect,ratio_list


def cal_gauges_merror(allRect):
    longs = []
    shorts = []
    for rect in allRect:
        longs.append(max(rect[1]))
        shorts.append(min(rect[1]))
    std_error = np.array((np.std(longs,ddof=1),
                          np.std(shorts,ddof=1)))*0.5*0.26
    longmean = np.mean(longs)
    shortmean = np.mean(shorts)
    long_huatu = (longs - longmean)*0.1
    short_huatu = (shorts - shortmean)*0.1
    cd = np.arange(len(longs))
    plt.figure()
    plt.plot(cd, long_huatu)
    plt.plot(short_huatu)
    plt.legend(["height","width"])
    return std_error
    

def translate_w_h(maRect):
    real_long = 6.71
    real_short = 3.97
    mmm = np.array(maRect[1])
    cal_long = max(mmm)
    cal_short = min(mmm)
    ratio = [real_long/cal_long, real_short/cal_short]
    return ratio
    

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    """
    前两个点是左上和右下,第三个点是阈值选取点,通过这个选点来识别
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img)
        LURD.append([x,y])
        
    
def save_contours_recs(imagePath,all_rect,four_corner):
    '''
    储存矩形四个点坐标以及最小外接矩形的中心(x,y),(宽度,高度),旋转角度
    '''
    this_name = imagePath.split('/')[-1].split('.')[0]
    np.savez('./colab_files/'+ this_name + '_corner.npz', *four_corner) # 解包list
    rectname = open('./colab_files/'+ this_name + '_rect.txt', 'w')
    # np.savez('./colab_files/'+ this_name + '_rect.npz', np.array(all_rect))
    for value in all_rect:
        for values in value:
            rectname.write(str(values))
            rectname.write(';')
        rectname.write('\n')
    rectname.close()
    return 


imagePath = './colab_files/lb2.bmp'
templatePath = './gauges/tmp.bmp'
#Loading image and template
img = cv2.imread(imagePath)

# choose gauges point 
LURD = []
cv2.namedWindow("image",0)
cv2.resizeWindow("image", 640, 512)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img)
    if cv2.waitKey(0)&0xFF==27:
        break
cv2.destroyAllWindows()

image = img

template = cv2.imread(templatePath)
#Converting template to grayscale.
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
rows,cols = template.shape[:2]
M = cv2.getRotationMatrix2D((rows/2,cols/2),180,1)
dst = cv2.warpAffine(template, M, (rows,cols))

#Applying Gaussian blur to image to reduce noise.
#blur_image = cv2.GaussianBlur(img,(3,3),0)

#Converting image to grayscale
blur_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
all_rect,ratio_list,std_error,all_labels=find_all_gauges(blur_image,img,LURD)

template_laplacian = cv2.Laplacian(template,cv2.CV_8U)
img_laplacian = cv2.Laplacian(blur_image,cv2.CV_8U)



