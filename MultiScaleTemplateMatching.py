import numpy as np
import cv2
import matplotlib.pyplot as plt

#Using custom template for Set A (setA.png)
#Using custom template for Set B (t2.png for t2_x.jpg, t3.png for t3_x.jpg )
#Use your template for set B (t1_x.jpg images)

def find_all_gauges(blurs,imgss):
    blur = blurs.copy()
    imgs = imgss.copy()
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th[th==0]=1
    th[th==255]=0
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(th,kernel,iterations = 1)
    num, labels = cv2.connectedComponents(erosion)
    x=[(i,np.sum(labels==i)) for i in range(num) 
       if np.sum(labels==i)>200 and np.sum(labels==i)<550]
    for i in range(len(x)):
        blur[labels==x[i][0]] = 255
    imgs[:,:,1] = blur
    plt.figure()
    plt.imshow(imgs)
    
    return x



imagePath = './capture_folder/save_img/l3.bmp'
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
find_all_gauges(blur_image,img)

template_laplacian = cv2.Laplacian(template,cv2.CV_8U)
img_laplacian = cv2.Laplacian(blur_image,cv2.CV_8U)
found = None

#Multi-Scaling Template
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
