#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from robotiq_interface import Robotiq
import multiprocessing as mp
import time
from datetime import datetime

cap_0 = cv2.VideoCapture(0)
cap_1 = cv2.VideoCapture(1)
fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
robotiq = Robotiq()
time.sleep(2)
robotiq.open_gripper()


out_0 = cv2.VideoWriter('output_0.avi', fourcc, 20.0, (640,480))
out_1 = cv2.VideoWriter('output_1.avi', fourcc, 20.0, (640,480))
run_once = True

while(cap_0.isOpened() and cap_1.isOpened()):
    ret_0, frame_0 = cap_0.read()
    ret_1, frame_1 = cap_1.read()
    if run_once:
    	robotiq.set_gripper_position(0)
    run_once = False    	
    if ret_0==True and ret_1==True:
        frame_0= cv2.flip(frame_0, 0)
        frame_1= cv2.flip(frame_1, 0)

        out_0.write(frame_0)
        out_1.write(frame_1)
        # cv2.imshow('frame_0', frame_0)
        # cv2.imshow('frame_1', frame_1)
        print ("Read grasping frame ...")

        if (robotiq.obj_dectected()):
			hold_frame_num = 0
			print ("Object holding ...")
			while (hold_frame_num < 100):
				print ("Read hold frame ...")
				hold_frame_num += 1
				ret_0, frame_0 = cap_0.read()
				ret_1, frame_1 = cap_1.read()
				frame_0= cv2.flip(frame_0, 0)
				frame_1= cv2.flip(frame_1, 0)
				out_0.write(frame_0)
				out_1.write(frame_1)
				# cv2.imshow('frame_0', frame_0)
				# cv2.imshow('frame_1', frame_1)
			print ("Work well!!!")
			break


			# if cv2.waitKey(1)&0xFF==ord('q'):  
			# 	break
    else:
        break
robotiq.open_gripper()
# reset camera
cap_0.release()
cap_1.release()
out_0.release()
out_1.release()
cv2.destroyAllWindows()



# if __name__ == '__main__':
# 	main()
