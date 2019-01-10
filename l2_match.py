# -*- coding: utf-8 -*-
# @Author: hcb
# @Date:   2018-09-13
# @Last Modified by:   hcb

import numpy as np
import cv2
import matplotlib.pyplot as plt
from coordinates_to_pc import getcenters
#from IPython import embed


maxidx = 100
input_name = ""
tmp_path = ""
# plt.rcParams['font.sans-serif']=['STSong'] 

def main():
    cap = cv2.VideoCapture(0)


    ret,img = cap.read()
    drawing, center1 = getcenters(np.array(img))
    cv2.imwrite("Match_{}_original.png".format(00), drawing)#the first good picture
    cv2.imshow("first", drawing)
    cv2.waitKey(0)
    counter = 1
    while counter<10:
        counter = counter+1
        ret,img = cap.read()
        drawing, center1  =getcenters(np.array(img))
    cv2.imwrite("Match_{}_original.png".format(00), drawing)#the first good picture
    cv2.imshow("nn", drawing)
    cv2.waitKey(0)
    temp_center= center1
    nn = len(temp_center)

    for idx in range(10):
        ret,img = cap.read()
        drawing, center2 = getcenters(np.array(img))
        #cv2.imwrite("Match_{}_original.png".format(idx), drawing)
        mm = len(center2)
        cv2.imshow("mm", drawing)
        cv2.waitKey(0)
        for i in range(nn):
            for j in range(mm):
                if np.sqrt(np.sum((temp_center[i] - center2[j]) ** 2))<30:
                    temp_center[i] = center2[j]
                    break
        img1 = np.zeros_like(img)
        for x in range(nn):
            cv2.arrowedLine(img1, tuple(center1[x]), tuple(temp_center[x]), (255, 255, 255), 4, tipLength=0.4)
        cv2.imwrite("Match_{}_shiliangtu.png".format(idx), img1)
    cap.release()
if __name__ == "__main__":
    main()
