from robotiq_interface import Robotiq
import time
import rospy
import numpy as np
import cv2
import os, sys
import multiprocessing as mp
from multiprocessing import Value
import json
from utils import ensure_dir, info

"""
class CamWorker(mp.Process):
    def __init__(self, cam_id, value, path):
        mp.Process.__init__(self)
        self.exit = mp.Event()
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.value = value
        self.path = path

    def __del__(self):
        self.shutdown()

    def run(self):
        idx = -1
        while not self.exit.is_set():
            idx = self.value.value
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 0)
            if idx >= 0:
                cv2.imwrite(os.path.join(self.path , '{0}.jpg'.format(idx)), frame)
                self.value.value = -1

    def shutdown(self):
        self.exit.set()
        self.cap.release()
"""

rospy.init_node('Robotiq_node', anonymous = True, disable_signals=False)
robotiq = Robotiq()

def collect_data(name, speed_seq=[0.2], force_seq=[50], times=1, manual=True):

    robotiq.open_gripper()
    datalog = {}

    # value0, value1 = Value('i', -1), Value('i', -1)
    dir0 = "/home/robot/Data/{}/cam0".format(name)
    dir1 = "/home/robot/Data/{}/cam1".format(name)
    ensure_dir(dir0)
    ensure_dir(dir1)

    # cap_0, cap_1 = CamWorker(0, value0, dir0), CamWorker(1, value1, dir1)
    # cap_0.start()
    # cap_1.start()
    cap_0, cap_1 = cv2.VideoCapture(0), cv2.VideoCapture(1)
    CAMLAG = 6

    for _ in range(CAMLAG):
        _, frame0 = cap_0.read()
        __, frame1 = cap_1.read()
    frame0, frame1 = cv2.flip(frame0, 0), cv2.flip(frame1, 0)
    cv2.imwrite("/home/robot/Data/{}/cam0/0_0.jpg".format(name), frame0)
    cv2.imwrite("/home/robot/Data/{}/cam1/0_0.jpg".format(name), frame1)

    robotiq.close_gripper(0.2, 1)
    time.sleep(2)
    for _ in range(CAMLAG):
        _, frame0 = cap_0.read()
        __, frame1 = cap_1.read()
    frame0, frame1 = cv2.flip(frame0, 0), cv2.flip(frame1, 0)
    robotiq.open_gripper()
    cv2.imwrite("/home/robot/Data/{}/cam0/0_1.jpg".format(name), frame0)
    cv2.imwrite("/home/robot/Data/{}/cam1/0_1.jpg".format(name), frame1)
    time.sleep(2)


    succ_grasp = 1
    # times = 5
    speed, force = 0, 0
    n = len(speed_seq)
    raw_input("Enter to start grasping...")
    for idx in range(n):
        speed, force = speed_seq[idx], force_seq[idx]
        for i in range(times):
            succ = False
            while not succ:
                # print ("Speed = {}, Force = {}, Times = {}. Press Enter to continue...".format(speed, force, i + 1))
                raw_input("Speed = {}, Force = {}, Times = {}. Press Enter to continue...".format(speed, force, i + 1))
                robotiq.close_gripper(speed, force)
                time.sleep(2)
                for _ in range(CAMLAG):
                    _, frame0 = cap_0.read()
                    __, frame1 = cap_1.read()
                frame0, frame1 = cv2.flip(frame0, 0), cv2.flip(frame1, 0)
                cv2.imwrite("/home/robot/Data/{}/cam0/{}.jpg".format(name, succ_grasp), frame0)
                cv2.imwrite("/home/robot/Data/{}/cam1/{}.jpg".format(name, succ_grasp), frame1)
                # value0.value, value1.value = succ_grasp, succ_grasp
                succ = robotiq.obj_dectected()
                raw_input()
                robotiq.open_gripper()
                time.sleep(2)
                if (not succ and manual):
                    '''
                    info("Contacted? (y/n)", domain="Manual")
                    # time.sleep(0.5)
                    # img = cv2.imread("/home/robot/Data/{}/cam0/{}.jpg".format(name, succ_grasp))
                    # cv2.imshow("{}_cam0".format(succ_grasp), img)
                    ch = sys.stdin.read(1)
                    while ch not in ('y', 'Y', 'n', 'N', '1', '0'):
                        print("Got bad character {}".format(ch))
                        info("Contacted? (y/n)", domain="Manual")
                        ch = sys.stdin.read(1)
                        ch = sys.stdin.read(1)
                    '''
                    ch = 'y'
                    succ |= (ch == 'y' or ch == 'Y' or ch == '1')
                    # ch = sys.stdin.read(1)

                if succ:
                    datalog[succ_grasp] = {
                        'speed': speed,
                        'force': force,
                        'index': i + 1
                    }
                    json.dump(datalog, open("/home/robot/Data/{}/data.json".format(name), 'w'))
                    succ_grasp += 1
                else:
                    info("No object detected")
    cap_0.release()
    cap_1.release()
    # cap_0.shutdown()
    # cap_1.shutdown()
    # cv2.destroyAllWindows()

def main():
    times = 1
    force_seq = [(0.5 * x) for x in range(1, 101)] # 5, 10, 15, ..., 100
    speed_seq = [0.2 for _ in range(1, 101)]

    name = None
    while name != "exit":
        name = raw_input("Input name of this object(type exit to stop): ")
        collect_data(name, speed_seq, force_seq, times, manual=True)

    info("Data collection stopped")

if __name__ == "__main__":
    main()
