# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'capture.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1493, 585)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pic_label = QtWidgets.QLabel(self.centralwidget)
        self.pic_label.setGeometry(QtCore.QRect(190, 10, 1280, 512))
        self.pic_label.setText("")
        self.pic_label.setObjectName("pic_label")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 160, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.capture_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.capture_btn.setObjectName("capture_btn")
        self.verticalLayout.addWidget(self.capture_btn)
        self.start_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.start_btn.setObjectName("start_btn")
        self.verticalLayout.addWidget(self.start_btn)
        self.stop_button = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.stop_button.setObjectName("stop_button")
        self.verticalLayout.addWidget(self.stop_button)
        self.moireButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.moireButton.setObjectName("moireButton")
        self.verticalLayout.addWidget(self.moireButton)
        self.reverse_cam_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.reverse_cam_btn.setObjectName("reverse_cam_btn")
        self.verticalLayout.addWidget(self.reverse_cam_btn)
        self.autoCali_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.autoCali_btn.setObjectName("autoCali_btn")
        self.verticalLayout.addWidget(self.autoCali_btn)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1493, 18))
        self.menubar.setObjectName("menubar")
        self.menuauto_exposure = QtWidgets.QMenu(self.menubar)
        self.menuauto_exposure.setObjectName("menuauto_exposure")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actioncam0_fringe = QtWidgets.QAction(MainWindow)
        self.actioncam0_fringe.setObjectName("actioncam0_fringe")
        self.actioncam1_fringe = QtWidgets.QAction(MainWindow)
        self.actioncam1_fringe.setObjectName("actioncam1_fringe")
        self.actioncam0_white = QtWidgets.QAction(MainWindow)
        self.actioncam0_white.setObjectName("actioncam0_white")
        self.actioncam1_white = QtWidgets.QAction(MainWindow)
        self.actioncam1_white.setObjectName("actioncam1_white")
        self.menuauto_exposure.addAction(self.actioncam0_fringe)
        self.menuauto_exposure.addAction(self.actioncam1_fringe)
        self.menuauto_exposure.addAction(self.actioncam0_white)
        self.menuauto_exposure.addAction(self.actioncam1_white)
        self.menubar.addAction(self.menuauto_exposure.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.capture_btn.setText(_translate("MainWindow", "Capture"))
        self.start_btn.setText(_translate("MainWindow", "Start Camera"))
        self.stop_button.setText(_translate("MainWindow", "Stop"))
        self.moireButton.setText(_translate("MainWindow", "Show moire image"))
        self.reverse_cam_btn.setText(_translate("MainWindow", "Reverse camera"))
        self.autoCali_btn.setText(_translate("MainWindow", "Auto calib"))
        self.menuauto_exposure.setTitle(_translate("MainWindow", "auto exposure"))
        self.actioncam0_fringe.setText(_translate("MainWindow", "cam0 fringe"))
        self.actioncam1_fringe.setText(_translate("MainWindow", "cam1 fringe"))
        self.actioncam0_white.setText(_translate("MainWindow", "cam0 white"))
        self.actioncam1_white.setText(_translate("MainWindow", "cam1 white"))
