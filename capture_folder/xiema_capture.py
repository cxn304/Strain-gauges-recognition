'''
author: cxn
version: 0.1.0
camera calibration
'''

import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from ximea import xiapi
from cwindow import Ui_MainWindow
from PIL import Image


class stripeWindow(QtWidgets.QMainWindow):
    '''
    新建一个window
    '''
    def __init__(self,img_path):
        super().__init__()
        self.setWindowTitle("Stripe Window")
        self.top = 0
        self.left = 0
        self.width = 1920
        self.height = 1080
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.self.moire_label=QtWidgets.QLabel()
        self.self.moire_label.resize(1920,1080)
        self.self.moire_label.show()
        self.self.moire_label.showFullScreen()
        img = QtGui.QPixmap(img_path).scaled(
            self.self.moire_label.width(), self.self.moire_label.height())
        self.self.moire_label.setPixmap(img)
        #self.close()

        

class Capture(QtWidgets.QMainWindow,Ui_MainWindow): # 这里名字要改
    def __init__(self,parent=None):
        super(Capture,self).__init__(parent)
        self.setupUi(self)
        self.slot_init()
        '''
        self.cam0 = xiapi.Camera(0)
        self.cam1 = xiapi.Camera(1)
        self.cam0sn = self.cam0.get_device_info_string('device_sn')
        self.cam1sn = self.cam1.get_device_info_string('device_sn')
        self.imgcount = 0
        self.timer = QtCore.QTimer()
        '''
    
        
        
    def slot_init(self):    # 这里有时候名称要改
        self.start_btn.clicked.connect(self.start_cap)
        self.capture_btn.clicked.connect(self.cap_shoot)
        self.stop_button.clicked.connect(self.stop_cap)
        self.moireButton.clicked.connect(self.show_Moire_imgs)
        
    
    def start_cap(self):
        self.cam0.open_device() # l
        self.cam1.open_device() # r
        self.cam0.set_exposure(17000)
        self.cam1.set_exposure(17000)
        self.img0 = xiapi.Image()
        self.img1 = xiapi.Image()
        self.cam0.start_acquisition()
        self.cam1.start_acquisition()
        self.startme()
        
        
    def startme(self):
        """
        开始按钮，启动计时器

        """
        self.frame_rate = 20
        rate = int(1000.0 / self.frame_rate)
        
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.timing_display)

        self.timer.start(rate)
        
        
    def timing_display(self):
        self.cam0.get_image(self.img0)
        self.cam1.get_image(self.img1)
        data0 = self.img0.get_image_data_numpy()    # l
        data1 = self.img1.get_image_data_numpy()
        imgShow = np.concatenate((data0,data1),axis=0)
        img_pil = Image.fromarray(imgShow)
        img_pix = img_pil.toqpixmap()
        self.pic_label.setPixmap(img_pix)
        self.pic_label.setScaledContents(True) # 使图片适应qlabel尺寸
        
        
    def cap_shoot(self):
        self.cam0.get_image(self.img0)
        self.cam1.get_image(self.img1)
        data0 = self.img0.get_image_data_numpy()
        data1 = self.img1.get_image_data_numpy()
        cv2.imwrite('./save_img/r' + str(self.imgcount) + '.bmp' 
                    , data0)
        cv2.imwrite('./save_img/l' + str(self.imgcount) + '.bmp' 
                    , data1)
        self.imgcount = self.imgcount + 1

        
        
    def stop_cap(self):
        self.timer.stop()
        self.cam0.stop_acquisition()
        self.cam0.close_device()
        self.cam1.stop_acquisition()
        self.cam1.close_device()
        self.imgcount = 0
        
        
    def show_Moire_imgs(self):
        '''
        打开8幅条纹图像并投影，投影间隙进行双目拍摄
        '''
        moire_img_index = 0     # 显示到第八幅图像要停
        all_img_path = './save_img/' # 条纹图像目录
        moire_images = os.listdir(all_img_path)
        self.moire_label=QtWidgets.QLabel()# 这个label一定要self
        self.moire_label.resize(1920,1080)
        self.moire_label.show()
        self.moire_label.showFullScreen()
        def update_image():
            # 更换图像用的嵌套函数
            nonlocal moire_img_index 
            def haha():
                # 用于singleShot的lambda函数,作用是延迟500ms后拍摄
                print(moire_images[moire_img_index])
            # nonlocal声明的变量不是局部变量,也不是全局变量,而是外部嵌套函数内的变量
            update_img_path = all_img_path + moire_images[moire_img_index]
            img = QtGui.QPixmap(update_img_path).scaled(
            self.moire_label.width(), self.moire_label.height())
            self.moire_label.setPixmap(img)
            QtCore.QTimer.singleShot(500, lambda:haha()) 
            # moire_img_index已经+1,说明等update_image执行完才执行singleShot
            moire_img_index += 1
            if moire_img_index >= 8:
                ttimer.stop()
                ttimer.deleteLater() # 清除自身
                del self.moire_label # 成功删除此label
                moire_img_index = 0
        
        ttimer = QtCore.QTimer()
        ttimer.timeout.connect(update_image)
        ttimer.start(2000)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # pool = mp.Pool(processes=4)
    ui = Capture()
    ui.show()
    sys.exit(app.exec_())
        



