import numpy as np
import cv2, pyclipper
import matplotlib.pyplot as plt

#Using custom template for Set A (setA.png)
#Using custom template for Set B (t2.png for t2_x.jpg, t3.png for t3_x.jpg )
#Use your template for set B (t1_x.jpg images)

def find_all_gauges(blurs,imgss):
    blur = blurs.copy()
    imgs = imgss.copy()
    row,col = blurs.shape[:2]
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th[th==0]=1
    th[th==255]=0
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(th,kernel,iterations = 1)
    plt.figure()
    plt.imshow(erosion)
    find_gauges = np.zeros((row,col),np.uint8)
    num, labels = cv2.connectedComponents(erosion)
    x=[(i,np.sum(labels==i)) for i in range(num) 
       if np.sum(labels==i)>200 and np.sum(labels==i)<500]
    for i in range(len(x)):
        find_gauges[labels==x[i][0]] = 1
    imgs[:,:,1] = blur
    # 寻找轮廓
    bimg, contours, hier = cv2.findContours(
        find_gauges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    canvas = np.copy(img)   # 声明画布拷贝自img
    ratio_list = [] # 每个图在图像中的长度和真实长度的比值,长和宽两个
    all_rect = []   # 每个矩形的信息
    for cidx,cnt in enumerate(contours):
        pco = pyclipper.PyclipperOffset()           # 多边形外扩函数
        pco.AddPath(cnt[:,0,:], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solu_contour = pco.Execute(2)   # 轮廓内缩或者外扩
        cnt_ = np.array(solu_contour)
        cnt__ = np.transpose(cnt_,(1,0,2))
        minAreaRect = cv2.minAreaRect(cnt__)  
        # 最小外接矩形的中心(x,y),(宽度,高度),旋转角度
        all_rect.append(minAreaRect)
        ratio_list.append(translate_w_h(minAreaRect))
        rectCnt = np.int64(cv2.boxPoints(minAreaRect))
        cv2.drawContours(canvas, [rectCnt], 0, (0,255,0), 3)
    plt.figure()
    plt.imshow(canvas)
    std_error = cal_gauges_merror(all_rect)
    
    return all_rect,ratio_list,std_error


def cal_gauges_merror(allRect):
    longs = []
    shorts = []
    for rect in allRect:
        longs.append(max(rect[1]))
        shorts.append(min(rect[1]))
    std_error = np.array((np.std(longs,ddof=1),np.std(shorts,ddof=1)))*0.5*0.26
    longmean = np.mean(longs)
    shortmean = np.mean(shorts)
    long_huatu = (longs - longmean)*0.26
    short_huatu = (shorts - shortmean)*0.26
    cd = np.arange(len(longs))
    plt.figure()
    p1 = plt.plot(cd, long_huatu)
    p2 = plt.plot(short_huatu)
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
    

imagePath = './capture_folder/save_img/li0.bmp'
templatePath = './gauges/tmp.bmp'


#Loading image and template
img = cv2.imread(imagePath)
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
all_rect,ratio_list,std_error = find_all_gauges(blur_image,img)

template_laplacian = cv2.Laplacian(template,cv2.CV_8U)
img_laplacian = cv2.Laplacian(blur_image,cv2.CV_8U)


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
    
    # Draw a bounding box around the detected result if greater than threshold.
    threshold = 0.8
    if(maxVal > threshold):
        print('Cursor Found!')
        cv2.circle(image, (int(maxLoc[0])+size[0]//2,
                           int(maxLoc[1])+size[1]//2), 3, (255, 0, 255), 3)
    else:
        print('Cursor not found')
        
    plt.imshow(image)
