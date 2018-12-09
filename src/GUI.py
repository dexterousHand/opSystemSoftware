import sys
import os
from os import path as op

import cv2
import numpy as np
import time
from utils import ensure_dir
import multiprocessing as mp
import temperature
from IPython import embed

try: # PyQt4
    from PyQt4 import QtGui
    from PyQt4.QtGui import (QWidget, QToolTip, QFont, QPushButton, QPixmap, QImage, QColor,
                            QSizePolicy, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit,
                            QComboBox, QDesktopWidget, QLineEdit, QIntValidator, QFileDialog,
                            QGridLayout, QMessageBox, QApplication, QPainter, QMainWindow, QSplitter, QFrame)
    from PyQt4.QtCore import QObject, QThread, pyqtSignal, QStringList, QString, QBasicTimer, Qt
except: # PyQt5
    from PyQt5.QtWidgets import (QWidget, QToolTip, QDesktopWidget, QSizePolicy, QProgressBar, QInputDialog, QLineEdit, QCheckBox,
                                 QPushButton, QApplication, QMainWindow, QAction, QTextEdit, QFileDialog, QComboBox, QLabel, QHBoxLayout, QVBoxLayout,
                                 QGridLayout, QSplitter, QFrame)
    from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, QObject, QBasicTimer, Qt
    from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage, QPainter
    QString = str

OFFLINE_FLAG = False

if not OFFLINE_FLAG:
    try: # Rospy
        from robotiq_interface import Robotiq
        import rospy
        rospy.init_node('Collector', anonymous=True)
        robotiq = Robotiq()
    except:
        print("Robotiq driver load failed. Continue without robot controller.")
        print("Is Robotiq driver running? Try execute: \n{}".format("roslaunch robotiq_85_bringup robotiq_85.launch"))
        robotiq = None
else:
    robotiq = None

CAMERA_PORT_0 = 2
CAMERA_PORT_1 = 1

"""

Structure:

    RecordVideo: Read camera and send signals. In GUI process. Waken up by timer
    ImageWidget: Show images. Used as slots.
    ForceWidget: Show force images. Used as slots.
    ImageSaver:  Save images. Used as slots.
    Pipe2Signal: Adapter. In a new QProcess started by GUI process. Takes Pipe<1> signals and send corresponding Qt signals to MainWidget.
    MainWidget:  GUI. Started by gui() process. 
                    -   Recv and process Qt signals from Pipe2Signal.
                    -   Send user commands from Pipe<1> to Pipe<2>.
                    -   Send click event signals to the classes above.

    gui(): Start GUI. Holds Pipe<1>. 
    robot(): Start robot controller. Holds Pipe<2>. Send robot status or response(signals) from Pipe<2> to Pipe<1>.

"""

class RecordVideo(QObject):
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        try:
            super().__init__(parent)
        except:
            super(RecordVideo, self).__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.timer = QBasicTimer()

    def start_recording(self):
        self.timer.start(20, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, image = self.camera.read()
        if read:
            self.image_data.emit(image)

    def stop_recording(self):
        self.timer.stop()


class ImageWidget(QWidget):
    def __init__(self, size=(320, 240), parent=None):
        try:
            super().__init__(parent)
        except:
            super(ImageWidget, self).__init__(parent)
        self.image = QImage()
        self.imgsize = size
        self.setFixedSize(*size)

    def image_data_slot(self, image_data):
        self.image = self.array2qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def array2qimage(self, image):
        assert isinstance(image, np.ndarray)
        image = cv2.resize(image, self.imgsize)
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

class ForceWidget(QWidget):
    # Cause signal shall be class object
    def __init__(self, size=(320, 240), parent=None):
        try:
            super().__init__(parent)
        except:
            super(ForceWidget, self).__init__(parent)
        self.image = QImage()
        self.imgsize = size
        self.setFixedSize(*size)

    def image_data_slot(self, image_data):
        self.image = self.array2qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def array2qimage(self, image):
        # TODO: Call force detector
        assert isinstance(image, np.ndarray)

        image = cv2.resize(image, self.imgsize)
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

class ImageSaver(QWidget):
    def __init__(self, size=(640, 480), idx=0, prefix="cam", parent=None):
        try:
            super().__init__(parent)
        except:
            super(ImageSaver, self).__init__(parent)
        self.imgsize = size
        self.path = "./data"
        ensure_dir(self.path)

        self.setidx(idx)
        self.prefix = prefix
        self.times_idx = 0
        self.counter = 0
        self.enabled = False

    def image_data_slot(self, image_data):
        if self.enabled:
            self.counter += 1
            # resize if needed
            cv2.imwrite(op.join(self.path, str(self.idx), str(self.times_idx), self.prefix, "{}.png".format(self.counter)), image_data)
        else:
            pass

    def update_times_idx(self):
        self.times_idx += 1
        ensure_dir(op.join(self.path, str(self.idx), str(self.times_idx)))
        ensure_dir(op.join(self.path, str(self.idx), str(self.times_idx), self.prefix))
        self.counter = 0

    def setidx(self, idx):
        self.idx = idx
        ensure_dir(op.join(self.path, str(self.idx)))

        names = os.listdir(op.join(self.path, str(self.idx)))
        names = list(filter(lambda x: x.isdigit(), names))
        names = list(map(int, names))
        self.times_idx = (max(names)) if len(names) > 0 else 0

    def enable(self):
        self.update_times_idx()
        self.enabled = True

    def disable(self):
        self.enabled = False

class Pipe2Signal(QThread):
    robot_signal = pyqtSignal(int)
    robot_heartbeat = pyqtSignal(str)
    def __init__(self, q2, parent=None):
        try:
            super().__init__(parent)
        except:
            super(Pipe2Signal, self).__init__(parent)
        self.q2 = q2

    def run(self):
        print("Pipe2Signal adapter started. pid = {}".format(os.getpid()))
        command = -1
        while command != 0:
            if isinstance(command, tuple):
                self.robot_heartbeat.emit(','.join(list(map(str, command))))
            elif isinstance(command, int) and command >= 1:
                print("Robot signal sent")
                self.robot_signal.emit(command)
            elif command is None or command == -1:
                pass # Initializing
            else:
                break
            command = self.q2.get(True)
        print("Pipe2Signal adapter stopped.")

class MainWidget(QWidget):
    def __init__(self, q1, q2, windowsize=(320, 240), usecam2=False, useforce=False, parent=None):
        try:
            super().__init__(parent)
        except:
            super(MainWidget, self).__init__(parent)
        self.setGeometry(800, 800, 800, 800)
        self.setWindowTitle('Robot Control Panel')

        self.q1 = q1   # Send to ROS
        self.adapter = Pipe2Signal(q2)  # Recv from ROS
        self.adapter.robot_signal.connect(self.adapter_handler)
        self.adapter.robot_heartbeat.connect(self.adapter_handler)

        self.usecam2 = usecam2
        self.useforce = useforce

        """
        4 Real-time displayers
        """

        self.image_widget_1 = ImageWidget(size=windowsize)
        self.record_video_1 = RecordVideo(camera_port=CAMERA_PORT_0)
        self.record_video_1.image_data.connect(self.image_widget_1.image_data_slot)
        self.image_saver_1 = ImageSaver(prefix="cam0")
        self.record_video_1.image_data.connect(self.image_saver_1.image_data_slot)

        if usecam2:
            self.image_widget_2 = ImageWidget(size=windowsize)
            self.record_video_2 = RecordVideo(camera_port=CAMERA_PORT_1)
            self.record_video_2.image_data.connect(self.image_widget_2.image_data_slot)
            self.image_saver_2 = ImageSaver(prefix="cam1")
            self.record_video_2.image_data.connect(self.image_saver_2.image_data_slot)
        else:
            self.image_widget_2 = QLabel("Cam2 Disabled")
            self.image_widget_2.setFixedSize(*windowsize)
            self.image_widget_2.setAlignment(Qt.AlignCenter)

        if useforce:
            self.force_widget_1 = ForceWidget(size=windowsize)
            self.record_video_1.image_data.connect(self.force_widget_1.image_data_slot)
            if usecam2:
                self.force_widget_2 = ForceWidget(size=windowsize)
                self.record_video_2.image_data.connect(self.force_widget_2.image_data_slot)
            else:
                self.force_widget_2 = QLabel("Force2 Disabled")
                self.force_widget_2.setFixedSize(*windowsize)
                self.force_widget_2.setAlignment(Qt.AlignCenter)

        """
        Prediction
        """
        self.prediction_1 = -1
        self.prediction_lbl_1 = QLabel("Pred_1: -1")
        self.prediction_2 = -1
        self.prediction_lbl_2 = QLabel("Pred_1: -1")

        """
        Command buttons & other labels
        """

        self.run_button = QPushButton('Open camera')
        self.stop_button = QPushButton('Close camera')
        self.run_button.clicked.connect(self.start_all) # self.record_video_1.start_recording)
        self.stop_button.clicked.connect(self.stop_all) # self.record_video_1.stop_recording)

        self.sample_idx = 0
        self.sample_idx_lbl = QLabel("Index: 0")
        self.sample_idx_edit = QLineEdit()
        self.sample_idx_edit.setText(QString("0"))
        self.sample_idx_set_btn = QPushButton("Set", self)
        self.sample_idx_set_btn.clicked.connect(self.set_sample_idx)

        self.sample_times = 0
        self.sample_times_lbl = QLabel("Times: 0")
        # TODO: Update this shit

        self.grasp_button = QPushButton("Grasp")
        self.grasp_button.clicked.connect(self.grasp)
        self.regrasp_button = QPushButton("Regrasp")
        self.regrasp_button.clicked.connect(self.regrasp)

        self.grasp20_button = QPushButton("Grasp 20x")
        self.grasp20_button.clicked.connect(self.grasp20)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_gripper)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_gripper)

        self.moving_flag = QLabel("Moving: False")
        self.detected_flag = QLabel("Detect: False")
        self.position_flag = QLabel("Position: N/A")
        self.current_flag = QLabel("Current: N/A")

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        self.init_ui()
        self.adapter.start()

        # self.action_response = False
        self.multitimes = 0


    def init_ui(self):
        """
        Layouts
        """
        hlines = [QFrame() for _ in range(4)]
        for hline in hlines:
            hline.setFrameShape(QFrame.HLine)
            hline.setFrameShadow(QFrame.Sunken)

        vlines = [QFrame() for _ in range(1)]
        for vline in vlines:
            vline.setFrameShape(QFrame.VLine)
            vline.setFrameShadow(QFrame.Sunken)

        def getimgbox(name, imgwidget):
            imgbox = QVBoxLayout()
            imgbox.addWidget(QLabel(name))
            imgbox.addWidget(imgwidget)
            return imgbox


        layout = QVBoxLayout()

        hbox1 = QHBoxLayout()
        hbox1.addStretch(1)
        hbox1.addLayout(getimgbox("Image1", self.image_widget_1))
        hbox1.addStretch(1)
        hbox1.addLayout(getimgbox("Image2", self.image_widget_2))
        hbox1.addStretch(1)

        vbox1 = QVBoxLayout()
        vbox1.addStretch(1)
        vbox1.addLayout(hbox1)
        vbox1.addStretch(1)
        vbox1.addWidget(hlines[0])
        vbox1.addStretch(1)

        if self.useforce:
            hbox2 = QHBoxLayout()
            hbox2.addStretch(1)
            hbox2.addLayout(getimgbox("Force1", self.force_widget_1))
            hbox2.addStretch(1)
            hbox2.addLayout(getimgbox("Force2", self.force_widget_2))
            hbox2.addStretch(1)
            vbox1.addLayout(hbox2)
            vbox1.addStretch(1)
        else:
            pass

        vbox2 = QGridLayout()
        vbox2.addWidget(self.sample_idx_lbl, 1, 1)
        vbox2.addWidget(self.sample_idx_edit, 2, 1, 2, 4)
        vbox2.addWidget(self.sample_idx_set_btn, 2, 5, 2, 6)
        vbox2.addWidget(self.sample_times_lbl, 4, 1, 4, 2)

        vbox3 = QGridLayout()
        vbox3.addWidget(QLabel("Predictions"), 1, 1)
        vbox3.addWidget(self.prediction_lbl_1, 3, 1)
        vbox3.addWidget(self.prediction_lbl_2, 4, 1)

        vbox4 = QGridLayout()
        vbox4.addWidget(QLabel("Robot commands"), 1, 1, 1, 2)
        vbox4.addWidget(self.open_button, 3, 2)
        vbox4.addWidget(self.close_button, 4, 2)
        vbox4.addWidget(self.grasp_button, 5, 2)
        vbox4.addWidget(self.regrasp_button, 6, 2)
        vbox4.addWidget(self.grasp20_button, 7, 2)
        vbox4.addWidget(QLabel(""), 6, 3)
        vbox4.addWidget(self.moving_flag, 9, 2)
        vbox4.addWidget(self.detected_flag, 10, 2)
        vbox4.addWidget(self.position_flag, 11, 2)
        vbox4.addWidget(self.current_flag, 12, 2)



        vbox5 = QVBoxLayout()
        vbox5.addStretch(1)
        vbox5.addLayout(vbox2)
        vbox5.addStretch(1)
        vbox5.addWidget(hlines[1])
        vbox5.addStretch(1)
        vbox5.addLayout(vbox3)
        vbox5.addStretch(1)
        vbox5.addWidget(hlines[2])
        vbox5.addStretch(1)
        vbox5.addLayout(vbox4)
        vbox5.addStretch(1)

        hbox6 = QHBoxLayout()
        hbox6.addStretch(1)
        hbox6.addLayout(vbox1)
        hbox6.addStretch(1)
        hbox6.addWidget(vlines[0])
        hbox6.addStretch(1)
        hbox6.addLayout(vbox5)
        hbox6.addStretch(1)

        hbox7 = QHBoxLayout()
        hbox7.addStretch(2)
        hbox7.addWidget(self.run_button)
        hbox7.addStretch(1)
        hbox7.addWidget(self.stop_button)
        hbox7.addStretch(1)
        hbox7.addWidget(self.exit_button)
        hbox7.addStretch(2)

        layout.addStretch(1)
        layout.addLayout(hbox6)
        layout.addStretch(1)
        layout.addWidget(hlines[3])
        layout.addStretch(1)
        layout.addLayout(hbox7)
        layout.addStretch(1)

        self.setLayout(layout)


    def closeEvent(self, event):
        self.deleteLater()

    def start_all(self):
        self.record_video_1.start_recording()
        if self.usecam2:
            self.record_video_2.start_recording()

    def stop_all(self):
        self.record_video_1.stop_recording()
        if self.usecam2:
            self.record_video_2.stop_recording()

    def set_sample_idx(self):
        self.sample_idx = int(self.sample_idx_edit.text())
        self.sample_idx_lbl.setText("Index: {}".format(self.sample_idx))
        self.image_saver_1.setidx(self.sample_idx)
        if self.usecam2:
            self.image_saver_2.setidx(self.sample_idx)

        self.sample_times = self.image_saver_1.times_idx
        if self.usecam2:
            assert(self.sample_times == self.image_saver_2.times_idx)

    def adapter_handler(self, value):
        if isinstance(value, QString):    # Robot status heartbeat
            _value = str(value).split(',')
            assert(len(_value) == 4)
            self.moving_flag.setText("Moving: {}".format(_value[0]))
            self.detected_flag.setText("Detect: {}".format(_value[1]))
            self.position_flag.setText("Position: {0:.4f}".format(float(_value[2])))
            self.current_flag.setText("Current: {0:.4f}".format(float(_value[3])))
        else:                           # Robot action response
            if value == 1:      # Debug
                self.shutdown_savers()
            elif value == 2:    # Opened
                print("Gripper opened.")
            elif value == 3:    # Closed
                print("Gripper closed.")
            elif value == 4:    # Grasped
                self.shutdown_savers()
                # self.action_response = True
                if self.multitimes > 0:
                    self.multitimes -= 1
                    print("{} times left".format(self.multitimes))
                    self.grasp()
                else:
                    print("Grasping completed.")
            else:
                print("Adapter handler got undefined signal {} with type {}".format(value, type(value)))
                

    def grasp(self):
        # Sync
        assert (self.sample_times == self.image_saver_1.times_idx), "sample times = {}, image saver 1 time idx = {}".format(self.sample_times, self.image_saver_1.times_idx)
        if self.usecam2:
            assert (self.sample_times == self.image_saver_2.times_idx), "sample times = {}, image saver 2 time idx = {}".format(self.sample_times, self.image_saver_2.times_idx)

        self.image_saver_1.enable()
        if self.usecam2:
            self.image_saver_2.enable()
        # self.pipe.send(1)   # 1 when debugging
        self.q1.put(4)
        self.sample_times += 1
        self.sample_times_lbl.setText("Times: {}".format(self.sample_times))

    def grasp20(self):
        self.multitimes = 20
        self.q1.put(5)

    def open_gripper(self):
        self.q1.put(2)

    def close_gripper(self):
        self.q1.put(3)

    def regrasp(self):
        self.sample_times -= 1
        self.grasp()
        # TODO: Needs lots of tests here lol

    def shutdown_savers(self):
        self.image_saver_1.disable()
        if self.usecam2:
            self.image_saver_2.disable()

def gui(q1, q2):
    print("GUI process started. pid = {}".format(os.getpid()))
    app = QApplication(sys.argv)
    main_widget = MainWidget(q1, q2, usecam2=True, useforce=False)
    main_widget.show()
    ret = app.exec_()
    q1.put(0)  # Release robot
    print("GUI process stopped. Ret = {}".format(ret))

def robot(q1, q2):
    """
    0: Quit
    1: Pretend doing something (like sleep 3s, used in debugging multiprocessing)
    2: Open gripper
    3: Close gripper
    4: Grasp
    5: Grasp 20x
    """
    print("Robot process started. pid = {}".format(os.getpid()))
    command = -1
    while command != 0:
        if command == 1: 
            time.sleep(3) # TODO: @Robotiq
            q2.put(1)
        elif command == 2:
            if not OFFLINE_FLAG:
                robotiq.open_gripper()
                q2.put(2)
            else:
                pass
        elif command == 3:
            if not OFFLINE_FLAG:
                robotiq.close_gripper()
                q2.put(3)
            else:
                pass
        elif command == 4:
            if not OFFLINE_FLAG:
                robotiq.grasp()
                q2.put(4)
            else:
                pass
        elif command == 5:
            if not OFFLINE_FLAG:
                q2.put(4)
            else:
                pass
        elif command < 0:
            pass    # IDLE
        else:
            print("Illegal robot command: {}".format(command))

        if OFFLINE_FLAG:
            robot_status = (1, 1, 8.5, 10)  # Debug magic number
        else:
            robot_status = robotiq.robot_status()

        q2.put(robot_status)

        if not q1.empty():
            command = q1.get(True)
        else:
            command = -1
            time.sleep(0.05)
    q2.put(0)  # Release GUI adapter
    print("Robot process stopped")

def main():
    print("Main process started. pid = {}".format(os.getpid()))
    # (con1, con2) = mp.Pipe()
    q1, q2 = mp.Queue(), mp.Queue()
    # q1: con1 to con2, gui send, robot get
    # q2: con2 to con1, robot send, gui get
    # guiproc = mp.Process(target=gui, args=(con1, ))
    guiproc = mp.Process(target=gui, args=(q1, q2, ))
    
    guiproc.start()

    # robotproc = mp.Process(target=robot, args=(con2, ))
    # robotproc.start()

    # robotproc.join()
    robot(q1, q2)
    guiproc.join()
    print("Main process stopped.")


if __name__ == '__main__':
    main()
