
from os import path as op
import sys

import json
import cv2
import numpy as np
import time
from IPython import embed
# import rospy

try: # PyQt4
    from PyQt4 import QtGui
    from PyQt4.QtGui import (QWidget, QToolTip, QFont, QPushButton, QPixmap, QImage, QColor,
                            QSizePolicy, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit,
                            QComboBox, QDesktopWidget, QLineEdit, QIntValidator, QFileDialog,
                            QGridLayout, QMessageBox, QApplication)
    from PyQt4.QtCore import QObject, QThread, pyqtSignal, QStringList, QString
except: # PyQt5
    from PyQt5.QtWidgets import (QWidget, QToolTip, QDesktopWidget, QSizePolicy, QProgressBar, QInputDialog, QLineEdit, QCheckBox,
                                 QPushButton, QApplication, QMainWindow, QAction, QTextEdit, QFileDialog, QComboBox, QLabel, QHBoxLayout, QVBoxLayout,
                                 QGridLayout)
    from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, QObject
    from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage
    QString = str
from utils import ensure_dir

# from robotiq_interface import Robotiq

try: # OpenCV 2
    fourcc = cv2.CV_FOURCC('M','J','P','G')
except: # OpenCV 3
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')


# robotiq = Robotiq()

class robotiqGUI(QWidget):
    video_controller = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # rospy.init_node('GUI', anonymous=True)
        self.record_video = False
        self.init_ui()

    def init_ui(self):
        """

        """
        self.setGeometry(800, 800, 800, 800)
        self.setWindowTitle('Robotiq Control Panel')
        # self.center()

        self.grid_label = QGridLayout()

        self.img_cam0 = QLabel()
        self.img_cam1 = QLabel()
        self.cam0_framenum = 0
        self.cam1_framenum = 0
        # self.img_cam2 = QLabel()

        self.lbl_name = QLineEdit()
        self.name = ""
        self.btn_save_name = QPushButton("Confirm", self)
        self.btn_save_name.clicked.connect(self.set_name)

        # Set image/video saving path
        self.save_path = QString(op.join(op.dirname(op.realpath(__file__)), "data"))
        self.lbl_save_path = QLineEdit()
        self.lbl_save_path.setText(self.save_path)
        self.btn_save_path = QPushButton("Set path", self)
        self.btn_save_path.clicked.connect(self.set_save_path)

        # Load model
        self.model_path = ""
        self.btn_load_model = QPushButton("Load model", self)
        self.btn_load_model.clicked.connect(self.set_load_model)

        # Conduct grasping
        self.btn_grasp = QPushButton("Grasp", self)
        self.btn_grasp.clicked.connect(self.set_grasp)

        # Exit
        self.btn_exit = QPushButton("Exit", self)
        self.btn_exit.clicked.connect(self.close)

        # self.btn_open_gripper = QPushButton("Open gripper", self)
        # self.btn_open_gripper.clicked.connect(robotiq.open_gripper)

        # self.btn_close_gripper = QPushButton("Close gripper", self)
        # self.btn_close_gripper.clicked.connect(robotiq.close_gripper)

        main_vbox = QVBoxLayout()

        def add_labelled_window(label, widget):
            layout = QVBoxLayout()
            layout.addStretch(5)
            layout.addWidget(QLabel(label))
            layout.addStretch(1)
            layout.addWidget(widget)
            layout.addStretch(5)
            return layout

        hbox1 = QHBoxLayout()
        vbox1 = add_labelled_window("Cam0", self.img_cam0)
        vbox2 = add_labelled_window("Cam1", self.img_cam1)
        hbox1.addStretch(1)
        hbox1.addLayout(vbox1)
        hbox1.addStretch(1)
        hbox1.addLayout(vbox2)
        hbox1.addStretch(1)

        hbox4 = QHBoxLayout()
        hbox4.addStretch(2)
        hbox4.addWidget(QLabel("Name: "))
        hbox4.addStretch(1)
        hbox4.addWidget(self.lbl_name)
        hbox4.addStretch(1)
        hbox4.addWidget(self.btn_save_name)
        hbox4.addStretch(2)

        hbox5 = QHBoxLayout()
        hbox5.addStretch(2)
        hbox5.addWidget(self.btn_grasp)
        hbox5.addStretch(1)
        hbox5.addWidget(self.btn_load_model)
        hbox5.addStretch(1)
        hbox5.addWidget(self.btn_save_path)
        hbox5.addStretch(1)
        hbox5.addWidget(self.btn_exit)
        hbox5.addStretch(2)
        
        # hbox6 = QHBoxLayout()
        # hbox6.addStretch(2)
        # hbox6.addWidget(self.btn_open_gripper)
        # hbox6.addStretch(2)
        # hbox6.addWidget(self.btn_close_gripper)
        # hbox6.addStretch(2)
        
        main_vbox.addStretch(1)
        main_vbox.addLayout(hbox1)
        main_vbox.addStretch(1)
        main_vbox.addLayout(hbox4)
        main_vbox.addStretch(1)
        main_vbox.addLayout(hbox5)
        main_vbox.addStretch(1)
        # main_vbox.addLayout(hbox6)
        # main_vbox.addStretch(1)

        self.setLayout(main_vbox)
        self.show()

    def set_name(self):
        self.name = self.lbl_name.text()

    def set_save_path(self):
        fname = QFileDialog.getExistingDirectory(self, "Set save path", op.dirname(op.abspath(__file__)))
        fname = str(fname)
        if not fname or not op.exists(fname) or not op.isdir(fname):
            return
        self.save_path = fname
        self.lbl_save_path.setText(QString(fname))

    def set_load_model(self):
        fname = QFileDialog.getOpenFileName(self, "Open model file", op.dirname(op.abspath(__file__)))
        fname = str(fname)
        if not fname or not op.exists(fname):
            return
        self.model_path = fname

    def set_grasp(self):
        pass
        
        # robotiq.open_gripper()

        self.video_controller.emit(op.join(self.save_path, self.name))
        # robotiq.close_gripper()
        # while not robotiq.obj_dectected():
        #     time.sleep(0.01)

        contactframes = self.cam0_framenum
        if self.record_video:
            while self.cam0_framenum - contactframes < 100:
                time.sleep(0.01)

        # robotiq.open_gripper()
        self.video_controller.emit("")
        

    
    def refresh_cam0(self, image):
        self.img_cam0.setPixmap(QPixmap.fromImage(image))
        self.cam0_framenum += 1
    
    def refresh_cam1(self, image):
        self.img_cam1.setPixmap(QPixmap.fromImage(image))
        self.cam1_framenum += 1
        

class ShowVideo(QThread):
    videoSignals0 = pyqtSignal(QImage)
    videoSignals1 = pyqtSignal(QImage)

    def __init__(self, cam=[0, 1], video=True, image=True):
        """
        Cam: [0, 1]
        """
        super().__init__()
        self.camera_port = cam
        self.cameras = [cv2.VideoCapture(x) for x in self.camera_port]
        self.run_video = 0

        self.video = video
        self.image = image

    def __del__(self):
        for x in self.cameras:
            x.release()

    def startVideo(self, path):
        if self.video:
            ensure_dir(path)
            self.outs = [cv2.VideoWriter(op.join(path, "{}.avi".format(x)), fourcc, 20.0, (640,480)) for x in range(len(self.cameras))]
        else:
            self.outs = None

        if self.image:
            imgdir = [op.join(path, "cam{}".format(x)) for x in range(len(self.cameras))]
            for x in imgdir:
                ensure_dir(x)

        writing_counter = 0
        self.run_video = 5
        while self.run_video > 0 and all([x.isOpened() for x in self.cameras]):
            rets = [cam.read() for cam in self.cameras]
            imgs = [cv2.resize(x[1], (640, 480)) if x[0] else None for x in rets]
            show_imgs = [cv2.resize(x[1], (320, 240)) if x[0] else None for x in rets]
            # imgs = [(cv2.flip(x[1], 0) if x[0] else None) for x in rets]
            
            color_swapped_images = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None) for img in show_imgs]
            qt_images = [(QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888) if img is not None else None) for img in color_swapped_images]

            # Refresh working cams
            if len(qt_images) > 0 and qt_images[0]:
                self.videoSignals0.emit(qt_images[0])
            if len(qt_images) > 1 and qt_images[1]:
                self.videoSignals1.emit(qt_images[1])

            if imgs[0] is not None:
                # self.outs[0].write(imgs[0])
                cv2.imwrite(op.join(imgdir[0], "{}_{}.png".format(0, writing_counter)), imgs[0])

            if all([img is not None for img in imgs]):
                writing_counter += 1
                if self.outs:
                    for i in range(len(imgs)):
                        self.outs[i].write(imgs[i])
                if self.image:
                    for i, dirname in enumerate(imgdir):
                        cv2.imwrite(op.join(dirname, "{}_{}.png".format(i, writing_counter)), imgs[i])
            self.run_video -= 1

        if self.outs is not None:
            for x in self.outs:
                x.release()

    def stopVideo(self):
        self.run_video = 0

    def switch(self, value):
        if value:
            self.startVideo(value)
        else:
            self.stopVideo()


def start_GUI():
    app = QApplication(sys.argv)
    gui = robotiqGUI()

    vids = ShowVideo(cam=[0], video=False, image=True)
    vids.videoSignals0.connect(gui.refresh_cam0)
    vids.videoSignals1.connect(gui.refresh_cam1)
    gui.video_controller.connect(vids.switch)
    vids.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    start_GUI()