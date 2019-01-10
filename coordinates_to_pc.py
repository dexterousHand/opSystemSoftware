#!/usr/bin/env python
# encoding: utf-8
#

import getpoint
import numpy as np
import cv2
import os
import os.path as op
from sklearn.cluster import KMeans
from time import time
import socket
import json
from sklearn.decomposition import PCA
maxidx = 50
input_name = ""
tmp_path = ""
g = getpoint.GetPoint()
def getcenters(img):
    bin_img = binarize(img)
    cir_img = getcircles(bin_img)
    drawing, centers= getcontours(cir_img)
    #cv2.imshow("cir_img", 255*cir_img)
    #cv2.waitKey(0)
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
    #global g

    b = np.zeros((img.shape[0], img.shape[1])).astype('uint8')
    ratio = img[:,:,0]/(img[:,:,2] + 1)
    b[ratio>2.5] = 1
    '''
    b1 = g.clf1.predict(img.reshape((-1,3)))
    b2 = g.clf2.predict(img.reshape((-1,3)))
    b3 = g.clf3.predict(img.reshape((-1,3)))
    b4 = g.clf4.predict(img.reshape((-1,3)))
    b = np.array(list(b1))

    b[b1 == 1 ] = 1
    b[b2 == 1 ] = 1
    b[b3 == 1 ] = 1
    b[b4 == 1 ] = 1
    b.resize(img.shape[0], img.shape[1])
    b = (b+1)/2
    b.astype('uint8')
    cv2.imshow("bb",255*b)
    '''

    return b

def getcircles(img):
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    jmg = cv2.dilate(img, el, iterations=1)
    jmg = cv2.erode(jmg, el, iterations=1)
    jmg = cv2.erode(jmg, el, iterations=1)
    jmg = cv2.dilate(jmg, el, iterations=1)
    cv2.imshow("jmg",jmg)
    cv2.waitKey(0)
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

    BUF_SIZE = 10240 #the size of buffer
    server_addr = ('166.111.138.104',8888) #combing ip and port number,geting address 
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)#produce a new object of socket
    client.connect(server_addr)#bind the address
    cap = cv2.VideoCapture(1)
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    for idx in range(maxidx):
        success, img = cap.read()
        print(img)
        #save("0-original",img)
        bin_img = binarize(img)
        #save("1-bin", bin_img * 255)
        cir_img = getcircles(bin_img)
        #save("2-erode", cir_img * 255)
        drawing, centers = getcontours(cir_img)
        #save("3-contour", drawing)
        client.sendall(repr(json.dumps(centers)))
        #data=client.recv(BUF_SIZE)#receive data from server
if __name__ == "__main__":
    main()
