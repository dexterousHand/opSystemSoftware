import sys
import os
from os import path as op

import cv2
import numpy as np
import time
from utils import ensure_dir
import multiprocessing as mp

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

"""

Structure:

    RecordVideo: Read camera and send signals. In GUI process. Wake up by timer
    ImageWidget: Show images. Used as slots.
    ForceWidget: Show force images. Used as slots.
    ImageSaver:  Save images. Used as slots.
    Pipe2Signal: Adapter. In a new QProcess started by GUI process. Takes Pipe<1> signals and send corresponding Qt signals to MainWidget.
    MainWidget:  GUI. Started by gui() process. 
                    -   Send user commands from Pipe<1> to Pipe<2>.
                    -   Send click event signals to the classes above.

    gui(): Start GUI. Pipe<1>. 
    robot(): Start robot controller. Pipe<2>. Send robot status from Pipe<2> to Pipe<1>.

"""

class RecordVideo(QObject):
    image_data = pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera_port = camera_port
        self.camera = cv2.VideoCapture(camera_port)
        self.timer = QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if self.camera_port == 0:
            img = cv2.imread("Match_5_original.png")
            self.image_data.emit(img)
        elif self.camera_port == 1:
            img = cv2.imread("Match_6_original.png")
            self.image_data.emit(img)
        """
        if (event.timerId() != self.timer.timerId()):
            return

        read, image = self.camera.read()
        if read:
            self.image_data.emit(image)
        """

    def stop_recording(self):
        self.timer.stop()


class ImageWidget(QWidget):
    def __init__(self, size=(320, 240), parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.imgsize = size
        self.setFixedSize(*size)


    def image_data_slot(self, image_data):
        self.image = self.array2qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()
        # cv2.imwrite("1.png", image_data)

    def array2qimage(self, image: np.ndarray):
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
    def __init__(self, size=(320, 240), idx=0, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.imgsize = size
        self.setFixedSize(*size)
        self.idx = idx

    def image_data_slot(self, image_data):
        self.image = self.array2qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def array2qimage(self, image: np.ndarray):
        # TODO: Call force detector
        if self.idx == 1:
            image = cv2.imread("Match_5.png")
        elif self.idx == 2:
            image = cv2.imread("Match_6.png")
        else:
            pass
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
        super().__init__(parent)
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
    def __init__(self, pipe, parent=None):
        super().__init__(parent)
        self.pipe = pipe

    def run(self):
        print("Pipe2Signal adapter started. pid = {}".format(os.getpid()))
        command = self.pipe.recv()
        while command != 0:
            if command == 1:
                print("Robot signal sent")
                self.robot_signal.emit(1)
            elif command != 0:
                print("Robot send undefined signal {} to GUI".format(command))
            else:
                break
            command = self.pipe.recv()
        print("Pipe2Signal adapter stopped.")

class MainWidget(QWidget):
    def __init__(self, pipe, windowsize=(320, 240), usecam2=False, useforce=False, parent=None):
        super().__init__(parent)
        self.setGeometry(800, 800, 800, 800)
        self.setWindowTitle('Robot Control Panel')

        self.pipe = pipe   # Send to ROS
        self.adapter = Pipe2Signal(pipe)  # Recv from ROS
        self.adapter.robot_signal.connect(self.adapter_handler)

        self.usecam2 = usecam2
        self.useforce = useforce

        """
        4 Real-time displayers
        """

        self.image_widget_1 = ImageWidget(size=windowsize)
        self.record_video_1 = RecordVideo(camera_port=0)
        self.record_video_1.image_data.connect(self.image_widget_1.image_data_slot)
        self.image_saver_1 = ImageSaver(prefix="cam0")
        self.record_video_1.image_data.connect(self.image_saver_1.image_data_slot)

        if usecam2:
            self.image_widget_2 = ImageWidget(size=windowsize)
            self.record_video_2 = RecordVideo(camera_port=1)
            self.record_video_2.image_data.connect(self.image_widget_2.image_data_slot)
            self.image_saver_2 = ImageSaver(prefix="cam1")
        else:
            self.image_widget_2 = QLabel("Cam2 Disabled")
            self.image_widget_2.setFixedSize(*windowsize)
            self.image_widget_2.setAlignment(Qt.AlignCenter)

        if useforce:
            self.force_widget_1 = ForceWidget(size=windowsize, idx=1)
            self.record_video_1.image_data.connect(self.force_widget_1.image_data_slot)
            if usecam2:
                self.force_widget_2 = ForceWidget(size=windowsize, idx=2)
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
        self.prediction_lbl_2 = QLabel("Pred_2: -1")

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

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        self.init_ui()
        self.adapter.start()


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
        vbox4.addWidget(self.grasp_button, 3, 2)
        vbox4.addWidget(self.regrasp_button, 4, 2)
        vbox4.addWidget(QLabel(""), 4, 3)

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
        if value == 1:
            self.undo_grasp()
        else:
            print("Adapter handler got undefined signal {}".format(value))

    def grasp(self):
        # Sync
        assert (self.sample_times == self.image_saver_1.times_idx)
        if self.usecam2:
            assert (self.sample_times == self.image_saver_2.times_idx)

        self.image_saver_1.enable()
        if self.usecam2:
            self.image_saver_2.enable()
        # Robotiq related functions. Not in this version
        self.pipe.send(1)
        self.sample_times += 1
        self.sample_times_lbl.setText("Times: {}".format(self.sample_times))

    def regrasp(self):
        # TODO: Without adding $times
        pass

    def undo_grasp(self):
        self.image_saver_1.disable()
        if self.usecam2:
            self.image_saver_2.disable()



def gui(pipe):
    print("GUI process started. pid = {}".format(os.getpid()))
    app = QApplication(sys.argv)
    # main_window = QWidget()
    main_widget = MainWidget(pipe, usecam2=True, useforce=True)
    # main_window.setCentralWidget(main_widget)
    # main_window.show()
    main_widget.show()
    ret = app.exec_()
    pipe.send(0)  # Release robot
    print("GUI process stopped. Ret = {}".format(ret))


def robot(pipe):
    """
    0: Quit
    1: Do something (sleep 3s)
    """
    print("Robot process started. pid = {}".format(os.getpid()))
    command = pipe.recv()
    while command != 0:
        if command == 1:
            time.sleep(3) # TODO: @Robotiq
            pipe.send(1)
        command = pipe.recv()
    pipe.send(0)  # Release GUI adapter
    print("Robot process stopped")


def main():
    print("Main process started. pid = {}".format(os.getpid()))
    (con1, con2) = mp.Pipe()
    guiproc = mp.Process(target=gui, args=(con1, ))
    guiproc.start()

    robotproc = mp.Process(target=robot, args=(con2, ))
    robotproc.start()

    robotproc.join()
    guiproc.join()
    print("Main process stopped.")


if __name__ == '__main__':
    main()
