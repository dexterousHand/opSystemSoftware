#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import os
import os.path as op
from sklearn.cluster import KMeans
from time import time
import socket
import json
maxidx = 50
input_name = ""
tmp_path = ""

def getcenters(img):
    bin_img = binarize(img)
    cir_img = getcircles(bin_img)
    drawing, centers= getcontours(cir_img)
    return drawing, centers

def getname(idx):
    global input_name, tmp_path
    #input_name = "input/input{}.jpg".format(idx)
    #tmp_path = "output/result_{}".format(input_name.split('/')[-1].split('.')[0])
    tmp_path = "output/result_{}".format(idx)

def save(s, img):
    cv2.imwrite(op.join(tmp_path, s + ".jpg"), img)

def getmeanstd(img, standard=None, threshold=0):
    t = img.copy().astype('float32')
    b, r = t[:,:,0], t[:,:,2]
    r[r == 0] = 1
    if standard is not None:
        mask = np.zeros_like(b).astype('bool')
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                mask[y, x] = max(abs(img[y, x] - standard)) <= threshold
    else:
        mask = np.ones_like(b).astype('bool')

    ratio = b / r # B / R
    return np.mean(ratio[mask]), np.std(ratio[mask])

def binarize(img):
    #standard = np.array([165, 188, 190])
    #threshold = 40
    #mean, std = getmeanstd(img)
    b = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
    ratio = img[:,:,0]/(img[:,:,2] + 1)
    b[ratio>2.5] = 1
    #for y in range(img.shape[0]):
    #    for x in range(img.shape[1]):
    #        if np.divide(img[y,x,0],img[y,x,2]+1) > 2.5:
    #            b[y, x] = 1
    return b

def getcircles(img):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    jmg = cv2.dilate(img, el, iterations=1)
    jmg = cv2.erode(jmg, el, iterations=1)
    jmg = cv2.erode(jmg, el, iterations=1)
    jmg = cv2.dilate(jmg, el, iterations=1)
    return jmg

def getcontours(img):
    drawing = img.copy()
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:

        m = cv2.moments(contour)
        if m['m00'] == 0:
            continue
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)

    centers = np.array(centers)

    for center in centers:
        cv2.circle(drawing, tuple(center), 3, 255, -1)
    return drawing, centers

def main():
    cap = cv2.VideoCapture(-1)
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    success, img = cap.read()
    cv2.imshow("img", img)
    cv2.waitKey(0)
    #save("0-original",img)
    bin_img = binarize(img)
    cv2.imshow("bin_img", bin_img * 255)
    cv2.waitKey(0)
    #save("1-bin", bin_img * 255)
    cir_img = getcircles(bin_img)
    cv2.imshow("cir_img", cir_img * 255)
    cv2.waitKey(0)
    #save("2-erode", cir_img * 255)
    drawing, centers = getcontours(cir_img)
    cv2.imshow("drawing", drawing * 255)
    cv2.waitKey(0)
    #save("3-contour", drawing)
    #client.sendall(repr(json.dumps(centers)))
    #data=client.recv(BUF_SIZE)#receive data from server
if __name__ == "__main__":
    main()
