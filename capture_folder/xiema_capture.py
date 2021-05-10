'''
author: cxn
version: 0.1.0
camera calibration
'''

import numpy as np
import cv2, os, sys,time,math
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from ximea import xiapi
from cwindow import Ui_MainWindow
from PIL import Image
  

class Capture(QtWidgets.QMainWindow,Ui_MainWindow): # 这里名字要改
    def __init__(self,parent=None):
        super(Capture,self).__init__(parent)
        self.setupUi(self)
        self.cam0 = xiapi.Camera(0)
        self.cam1 = xiapi.Camera(1)
        self.cam0sn = self.cam0.get_device_info_string('device_sn')
        self.cam1sn = self.cam1.get_device_info_string('device_sn')
        self.imgcount = 0
        self.timer = QtCore.QTimer()
        self.slot_init()
        self.cam0_white_exposure = 8000
        self.cam1_white_exposure = 8000
        
        
    def slot_init(self):    # 这里有时候名称要改
        self.start_btn.clicked.connect(self.start_cap)
        self.capture_btn.clicked.connect(self.cap_shoot)
        self.stop_button.clicked.connect(self.stop_cap)
        self.moireButton.clicked.connect(self.show_Moire_imgs)
        self.reverse_cam_btn.clicked.connect(self.reverse_camera)
        self.autoCali_btn.clicked.connect(self.auto_cali)
        self.actioncam0_white.triggered.connect(self.cam0_white)
        self.actioncam1_white.triggered.connect(self.cam1_white)
        
    
    def start_cap(self):
        self.cam0.open_device() # l
        self.cam1.open_device() # r
        self.cam0.set_exposure(7000)
        self.cam1.set_exposure(8000)
        self.img0 = xiapi.Image()
        self.img1 = xiapi.Image()
        self.cam0.start_acquisition()
        self.cam1.start_acquisition()
        self.startme()
        
        
    def cam0_white(self):
        '''
        调整相机0的在白光下的曝光,这个相机每次拍图片最大灰度似乎跳跃很大
        '''
        self.moire_label=QtWidgets.QLabel()# 这个label一定要self
        self.moire_label.resize(1920,1080)
        self.moire_label.show()
        self.moire_label.showFullScreen()
        img = QtGui.QPixmap('./fringe_example/z.png').scaled(
                self.moire_label.width(), self.moire_label.height())
        self.moire_label.setPixmap(img)
        def update_exposure():
            self.cam0.set_exposure(self.cam0_white_exposure)
            self.cam0.get_image(self.img0)
            data0 = self.img0.get_image_data_numpy()
            max_pixel = np.max(data0)
            if max_pixel>250:
                self.cam0_white_exposure -= 50
            elif max_pixel < 240:
                self.cam0_white_exposure += 50
            else:
                ttimer.stop()
                ttimer.deleteLater() # 清除自身
                del self.moire_label # 成功删除此label
            print('cam0 exposure is: ' + str(self.cam0_white_exposure))
            print(max_pixel)
        ttimer = QtCore.QTimer()
        ttimer.timeout.connect(update_exposure)
        ttimer.start(700)         
      
        
    def cam1_white(self):
        '''
        调整相机1的在白光下的曝光
        '''
        self.moire_label=QtWidgets.QLabel()# 这个label一定要self
        self.moire_label.resize(1920,1080)
        self.moire_label.show()
        self.moire_label.showFullScreen()
        img = QtGui.QPixmap('./fringe_example/z.png').scaled(
                self.moire_label.width(), self.moire_label.height())
        self.moire_label.setPixmap(img)
        def update_exposure():
            self.cam1.set_exposure(self.cam1_white_exposure)
            self.cam1.get_image(self.img0)
            data0 = self.img0.get_image_data_numpy()
            max_pixel = np.max(data0)
            if max_pixel>250:
                self.cam1_white_exposure -= 50
            elif max_pixel < 240:
                self.cam1_white_exposure += 50
            else:
                ttimer.stop()
                ttimer.deleteLater() # 清除自身
                del self.moire_label # 成功删除此label
            print('cam0 exposure is: ' + str(self.cam1_white_exposure))
            print(max_pixel)
        ttimer = QtCore.QTimer()
        ttimer.timeout.connect(update_exposure)
        ttimer.start(700) 
        
        
    def startme(self):
        """
        开始按钮，启动计时器

        """
        self.frame_rate = 30
        rate = int(1000.0 / self.frame_rate)
        
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.timing_display)

        self.timer.start(rate)
        
        
    def timing_display(self):
        '''
        实时显示30帧
        '''
        self.cam0.get_image(self.img0)
        self.cam1.get_image(self.img1)
        data0 = self.img0.get_image_data_numpy()    # l
        data1 = self.img1.get_image_data_numpy()
        imgShow = np.concatenate((data0,data1),axis=1)
        img_pil = Image.fromarray(imgShow)
        img_pix = img_pil.toqpixmap()
        self.pic_label.setPixmap(img_pix)
        self.pic_label.setScaledContents(True) # 使图片适应qlabel尺寸
        
        
    def cap_shoot(self):
        '''
        拍摄双目图像,通过点击
        '''
        self.cam0.get_image(self.img0)
        self.cam1.get_image(self.img1)
        data0 = self.img0.get_image_data_numpy()
        data1 = self.img1.get_image_data_numpy()
        cv2.imwrite('./save_img/l' + str(self.imgcount) + '.png' 
                    , data0)
        cv2.imwrite('./save_img/r' + str(self.imgcount) + '.png' 
                    , data1)
        self.imgcount = self.imgcount + 1

              
    def stop_cap(self):
        self.timer.stop()
        self.cam0.stop_acquisition()
        self.cam0.close_device()
        self.cam1.stop_acquisition()
        self.cam1.close_device()
        self.imgcount = 0
        
        
    def capture_moire(self):
        '''
        依次触发此函数拍摄moire图像,拍摄顺序非常重要,这里规定先拍横条纹,再拍竖条纹
        再拍十字架,最后拍白图
        '''
        time_name = str(math.floor(time.time()*10)) # 返回当前的0.1秒数
        self.cam0.get_image(self.img0)
        self.cam1.get_image(self.img1)
        data0 = self.img0.get_image_data_numpy()
        data1 = self.img1.get_image_data_numpy()
        cv2.imwrite('./moire_img/'+ self.folder_name+'/l_' + time_name+'.png' 
                    , data0)
        cv2.imwrite('./moire_img/'+ self.folder_name+'/r_' + time_name+'.png' 
                    , data1)
        
        
    def reverse_camera(self):
        '''
        调换左右相机位置
        '''
        tmp = self.cam0
        self.cam0 = self.cam1
        self.cam1 = tmp
        
        
    def show_Moire_imgs(self):
        '''
        打开8x3幅条纹图像并逐次投影,投影后500ms进行双目拍摄
        '''
        #self.timer.stop()   # 此时停止以30帧每秒采集图像
        self.cam0.set_exposure(7000)    # 投条纹图的曝光
        self.cam1.set_exposure(8000)
        self.folder_name = str(math.floor(time.time()*10)) # 新建文件夹名字
        os.mkdir('./moire_img/'+self.folder_name)
        moire_img_index = 0
        all_img_path = './fringe_example/' # 要投影的条纹图像目录
        moire_images = os.listdir(all_img_path)
        example_imgs_num = len(moire_images)
        self.moire_label=QtWidgets.QLabel()# 这个label一定要self
        self.moire_label.resize(1920,1080)
        self.moire_label.show()
        self.moire_label.showFullScreen()
        def update_image():
            # 更换图像用的嵌套函数
            nonlocal moire_img_index 
            # nonlocal声明的变量不是局部变量,也不是全局变量,而是外部嵌套函数内的变量
            if moire_img_index < example_imgs_num:
                if moire_img_index==example_imgs_num-2:  #  动态调整曝光
                    self.cam0.set_exposure(self.cam0_white_exposure)
                    self.cam1.set_exposure(self.cam1_white_exposure)
                update_img_path = all_img_path + moire_images[moire_img_index]
                img = QtGui.QPixmap(update_img_path).scaled(
                self.moire_label.width(), self.moire_label.height())
                self.moire_label.setPixmap(img)
                # moire_img_index已经+1,说明等update_image执行完才执行singleShot
                # 连续拍摄2张求平均以保证没有噪声
                QtCore.QTimer.singleShot(500, lambda:self.capture_moire())
                QtCore.QTimer.singleShot(1000, lambda:self.capture_moire())
                QtCore.QTimer.singleShot(1500, lambda:self.capture_moire())
                moire_img_index += 1
            else:
                ttimer.stop()
                ttimer.deleteLater() # 清除自身
                del self.moire_label # 成功删除此label
                #rate = int(1000.0 / self.frame_rate)
                #self.timer.start(rate)
                
     
        ttimer = QtCore.QTimer()
        ttimer.timeout.connect(update_image)
        ttimer.start(2000)  # 两秒拍一次


    def auto_cali(self):
        '''
        自动标定,间隔几秒拍一次
        '''
        imgs_num = 0
        self.cam0.set_exposure(self.cam0_white_exposure+1000)
        self.cam1.set_exposure(self.cam1_white_exposure+1000)
        self.moire_label=QtWidgets.QLabel()# 这个label一定要self
        self.moire_label.resize(1920,1080)
        self.moire_label.show()
        self.moire_label.showFullScreen()
        img = QtGui.QPixmap('./fringe_example/z.png').scaled(
                self.moire_label.width(), self.moire_label.height())
        self.moire_label.setPixmap(img)
        def update_image():
            # 更换图像用的嵌套函数
            nonlocal imgs_num 
            QtCore.QTimer.singleShot(1000, lambda:self.cap_shoot())
            imgs_num += 1
            if imgs_num>20:
                ttimer.stop()
                ttimer.deleteLater() # 清除自身
                del self.moire_label # 成功删除此label
        
        ttimer = QtCore.QTimer()
        ttimer.timeout.connect(update_image)
        ttimer.start(2000)  # 两秒换一个标定板姿态
        return


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Capture()
    ui.show()
    sys.exit(app.exec_())
        



