
from PIL import Image
from PIL import ImageFilter
from math import sqrt
import numpy as np
import os
import cv2

class Tempdiscern():
    def __init__(self,r=8,R=11,E=14,a=255,b=0):
        self.r = r
        self.R = R
        self.E = E
        self.a = a
        self.b = b

        self.p_img = self.point_mask()



    def loadImage(self, filepath):
        img = Image.open(filepath)
        img = img.crop((80,60, 560,420))
        img.show()
        pix = img.load()
        return img, pix

    def get_degrees(self):
        self.n_img = Image.new("L", self.img.size)
        self.purple_img = self.purple_degree()
        #self.n_img.show()
        self.blue_img = self.blue_degree()
        #self.n_img.show()
        self.black_img = self.black_degree()
        #self.n_img.show()
        self.white_img = self.white_degree()
        #self.n_img.show()



    def white_degree(self):
        threshold_1 = 105#105 150
        threshold_2 = 123#123 180
        n_img = self.n_img
        n_pix = n_img.load()
        pix = self.img.load()
        white_re = np.zeros((self.img.size[0],self.img.size[1]))
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                if pix[i, j][0]+ pix[i, j][1] +pix[i, j][2] < 560: #600
                    n_pix[i, j] = (0,)
                    white_re[i, j] = -1
                    continue
                a = pow(pix[i, j][0] - pix[i, j][1], 2) + pow(pix[i, j][0] - pix[i, j][2], 2) \
                    + pow(pix[i, j][1] - pix[i, j][2], 2)
                if (a < threshold_1):
                    n_pix[i, j] = (255,)
                    white_re[i,j] = 1
                elif (a < threshold_2):#123
                    n_pix[i, j] = (128,)
                    white_re[i,j] = 0
                else:
                    n_pix[i, j] = (0,)
                    white_re[i, j] = -1
        #n_img.show("white_degree")
        #print("white_degree is ok")
        return white_re



    def blue_degree(self):
        threshold_1 = 8
        threshold_2 = 15
        n_img = self.n_img
        n_pix = n_img.load()
        pix = self.img.load()
        blue_re = np.zeros((self.img.size[0], self.img.size[1]))
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                if pix[i, j][2] < 190 or pix[i,j][0] + pix[i,j][1] > 320:
                    n_pix[i, j] = (0,)
                    blue_re[i, j] = -1
                    continue
                a = pix[i,j][1] - 1.11*pix[i,j][0] - 10.35
                if (a < threshold_1 and pix[i,j][2] > 195):
                    n_pix[i, j] = (255,)
                    blue_re[i, j] = 1
                elif (a < threshold_2 and pix[i,j][2] > 190):
                    n_pix[i, j] = (128,)
                    blue_re[i, j] = 0
                else:
                    n_pix[i, j] = (0,)
                    blue_re[i, j] = -1
        #n_img.show("blue_degree")
        #print("blue_degree is ok")
        return blue_re

    def black_degree(self):
        threshold_1 = 5
        threshold_2 = 10
        n_img = self.n_img
        n_pix = n_img.load()
        pix = self.img.load()
        black_re = np.zeros((self.img.size[0], self.img.size[1]))
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                if  pix[i, j][0] + pix[i, j][1] + pix[i, j][2]> 260:
                    n_pix[i, j] = (0,)
                    black_re[i, j] = -1
                    continue
                a = pix[i, j][1] - 0.617 * pix[i, j][0] - 18.47
                if (a < threshold_1):
                    n_pix[i, j] = (255,)
                    black_re[i, j] = 1
                elif (a < threshold_2):
                    n_pix[i, j] = (128,)
                    black_re[i, j] = 0
                else:
                    n_pix[i, j] = (0,)
                    black_re[i, j] = -1
        #n_img.show("black_degree")
        #print("black degree is ok")
        return black_re

    def purple_degree(self):
        threshold_1 = 5
        threshold_2 = 10
        n_img = self.n_img
        n_pix = n_img.load()
        pix = self.img.load()
        purple_re = np.zeros((self.img.size[0], self.img.size[1]))
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                if pix[i, j][2] < 160 or pix[i, j][2] > 200:
                    n_pix[i, j] = (0,)
                    purple_re[i, j] = -1
                    continue
                a = pix[i, j][1] - 0.934 * pix[i, j][0] + 0.184
                if (a < threshold_1 ):
                    n_pix[i, j] = (255,)
                    purple_re[i, j] = 1
                elif (a < threshold_2):
                    n_pix[i, j] = (128,)
                    purple_re[i, j] = 0
                else:
                    n_pix[i, j] = (0,)
                    purple_re[i, j] = -1
        #n_img.show("purple_degree")
        #print("purple degree is ok")
        return purple_re


    def point_mask(self):
        pa = 1
        pb = -1
        n_img = Image.new("L", (2 * self.E + 1, 2 * self.E + 1))
        n_pix = n_img.load()
        center_x = 14
        center_y = 14
        point_re = np.zeros((n_img.size[0],n_img.size[1]))
        for i in range(n_img.size[0]):
            for j in range(n_img.size[1]):
                disqure = pow(i - center_x, 2) + pow(j - center_y, 2)
                if disqure < self.r * self.r:
                    n_pix[i, j] = (self.a,)
                    point_re[i,j] = pa
                elif disqure < self.R * self.R:
                    n_pix[i, j] = int((self.b * (disqure - self.r * self.r) + self.a * (self.R * self.R - disqure)) / (self.R * self.R - self.r * self.r))
                    point_re[i, j] = (pb * (disqure - self.r * self.r) + pa * (self.R * self.R - disqure)) / (self.R * self.R - self.r * self.r)
                elif disqure < self.E * self.E:
                    n_pix[i, j] = (self.b,)
                    point_re[i, j] = pb
                else:
                    n_pix[i, j] = 0
        #n_img.show("point_mask")

        #print("point_mask is ok")
        #print(point_re)
        return point_re

    def likeness(self, center_x, center_y, color):
        x = center_x
        y = center_y
        E = self.E
        if color == "white":
            white_part = self.white_img[x-E:x+E+1, y-E:y+E+1]
            re_np = self.p_img*white_part
        elif color == "blue":
            blue_part = self.blue_img[x - E:x + E + 1, y - E:y + E + 1]
            re_np = self.p_img * blue_part
        elif color == "purple":
            purple_part = self.purple_img[x - E:x + E + 1, y - E:y + E + 1]
            re_np = self.p_img * purple_part
        elif color == "black":
            black_part = self.black_img[x - E:x + E + 1, y - E:y + E + 1]
            re_np = self.p_img * black_part
        else:
            print("parameter color in likeness is wrong")
        # print("    --------------- - - - - -  - - -- -  - - - - - ")
        # print(re_np.size)
        # print(type(re_np.sum()))
        return re_np.sum()


    def discern(self, img):
        self.img = Image.fromarray(cv2.cvtColor(img[60:420, 80:560], cv2.COLOR_BGR2RGB))
        self.img.show()
        self.pix = self.img.load()
        #self.img, self.pix = self.loadImage(filepath)
        # print("loadimage is ok")

        self.get_degrees()

        # n_img = Image.new("L", self.img.size)
        # n_pix = n_img.load()
        color_n = [0, 0, 0, 0]
        color_name = ["purple", "blue", "black", "white"]
        for x in range(self.img.size[0] - 2 * self.E):
            for y in range(self.img.size[1] - 2 * self.E):
                purple = self.likeness(x + self.E, y + self.E, "purple")
                blue = self.likeness(x + self.E, y + self.E, "blue")
                black = self.likeness(x + self.E, y + self.E, "black")
                white = self.likeness(x + self.E, y + self.E, "white")
                colors = [purple, blue, black, white]
                for i in range(4):
                    if colors[i] > 280:
                        color_n[i] += 1
        print(color_n)
        print(color_name[color_n.index(max(color_n))])

        print("            ")


#t = Tempdiscern()
#t.discern('55 (2).png')
'''
white: 111111/17/cam1/50.png, 222222/19/cam1/42.png
blue:  111111/9/cam1/50.png, 222222/12/cam1/52.png
black: black/BING_HEI/WIN_20181128_16_34_38_Pro.jpg, black/BING_HEI/WIN_20181128_16_33_26_Pro.jpg
purple: 111111/2/cam1/54.png 222222/2/cam1/59.png

'''

'''
print(img.size)
print(pix[391,251])
print(pix[391,268])
print(pix[400,260])
print(pix[391,260])
print(pix[389,256])
print(pix[391,261])
print(pix[397,262])
print(pix[387,263])
print(pix[389,255])
'''
'''
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if(pix[i,j][0]>173 & pix[i,j][0]<200 & pix[i,j][1]>197 & pix[i,j][1]<202 &pix[i,j][2]>199 & pix[i,j][2]<213):
            pix[i,j]=(0,0,0)
'''
#img.show()
#conF = img.filter(ImageFilter.CONTOUR)             ##找轮廓
#conF.show()



