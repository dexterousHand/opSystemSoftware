import sklearn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.covariance import  EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
class GetPoint():
    def __init__(self):

        c = cv2.imread("/home/robot/Desktop/black/1.png")
        d = cv2.imread("/home/robot/Desktop/black/2.png")
        e = cv2.imread("/home/robot/Desktop/black/3.png")
        f = cv2.imread("/home/robot/Desktop/black/4.png")
        g = cv2.imread("/home/robot/Desktop/black/5.png")

        b1 = cv2.imread("/home/robot/Desktop/blue/1.png")
        b2 = cv2.imread("/home/robot/Desktop/blue/2.png")
        b3 = cv2.imread("/home/robot/Desktop/blue/3.png")
        b4 = cv2.imread("/home/robot/Desktop/blue/2nd/9.png")
        b5 = cv2.imread("/home/robot/Desktop/blue/3nd/18.png")

        w1= cv2.imread("/home/robot/Desktop/white/4.png")
        w2 = cv2.imread("/home/robot/Desktop/white/5.png")
        w3 = cv2.imread("/home/robot/Desktop/white/6.png")
        w4 = cv2.imread("/home/robot/Desktop/white/2nd/8.png")
        w5 = cv2.imread("/home/robot/Desktop/white/3rd/23.png")

        p1= cv2.imread("/home/robot/Desktop/purple/12.png")
        p2= cv2.imread("/home/robot/Desktop/purple/13.png")
        p3= cv2.imread("/home/robot/Desktop/purple/14.png")



        blacks = []
        blues = []
        whites = []
        purples = []
        backgrounds = []
        for i in range(4):
            for j in range(4):
                backgrounds.append(c[106 + 3*i,189 + 3*j])
                backgrounds.append(c[243 + 3*i,255 + 3*j])
                backgrounds.append(c[138 + 3*i,120 + 3*j])
                backgrounds.append(c[274 + 3*i,186 + 3*j])

                backgrounds.append(b1[165 + 3*i,81 + 3*j])
                backgrounds.append(b1[278 + 3*i,392 + 3*j])

                backgrounds.append(b5[108 + 3*i,264 + 3*j])
                backgrounds.append(b5[240 + 3*i,123 + 3*j])

                backgrounds.append(b4[229 + 3*i,257 + 3*j])
                backgrounds.append(b4[330 + 3*i,388 + 3*j])
                backgrounds.append(b4[258 + 3*i,194 + 3*j])
                backgrounds.append(b4[324 + 3*i,192 + 3*j])

        for i in range(6):
            for j in range(6):
                blacks.append(c[128 + 34*i,175 + 34*j])
                blacks.append(d[131 + 34*i,170 + 34*j])
                blacks.append(e[131 + 34*i,170 + 34*j])
                blacks.append(f[131 + 34*i,170 + 34*j])
                blacks.append(g[130 + 34*i,175 + 34*j])

                blues.append(b1[128 + 34*i,175 + 34*j])
                blues.append(b2[125 + 34*i,175 + 34*j])
                blues.append(b3[125 + 34*i,174 + 34*j])
                blues.append(b4[154+ 32*i,186 + 32*j])
                blues.append(b5[162+ 34*i,183 + 34*j])

                whites.append(w1[189 + 35*i,213 + 35*j])
                whites.append(w2[125 + 34*i,175 + 34*j])
                whites.append(w3[125 + 34*i,174 + 34*j])
                whites.append(w4[167+ 32*i,194 + 32*j])
                whites.append(w5[174+ 35*i,196 + 35*j])

        for i in range(6):
            for j in range(9):
               purples.append(p1[157 + 32*i,186 + 32*j])
               purples.append(p2[186 + 32*i,152 + 32*j])
               purples.append(p3[187 + 32*i,149 + 32*j])

        #cv2.imshow("black",c)#34 35
        #cv2.waitKey(0)

        blacks = np.array(blacks)
        blues = np.array(blues)
        whites = np.array(whites)
        purples = np.array(purples)
        backgrounds = np.array(backgrounds)
        all = np.concatenate((blacks, blues, whites, purples, backgrounds), axis=0)
        #print(all.shape)
        pca = PCA(n_components=2)
        new_all = pca.fit_transform(all)
        #print(new_all.shape)
        plt.figure(figsize = (8,5))
        plt.scatter(new_all[0:180,0],new_all[0:180,1],s = 10, c = "r",marker='o')
        plt.scatter(new_all[180:360,0],new_all[180:360,1],s = 10,c = "b",marker='v')
        plt.scatter(new_all[360:540,0],new_all[360:540,1],s = 10,c = "y",marker='s')
        plt.scatter(new_all[540:702,0],new_all[540:702,1],s = 10,c = "g",marker='p')
        plt.scatter(new_all[702:894,0],new_all[702:894,1],s = 10,c = "m",marker='p')
        #plt.colorbar()
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("e")
        #plt.show()

        self.clf1 = svm.OneClassSVM(nu = 0.01, kernel="rbf", gamma= 0.1)
        self.clf1.fit(blacks)



        self.clf2 = svm.OneClassSVM(nu = 0.01, kernel="poly", gamma= 0.1)
        self.clf2.fit(blues)


        self.clf3 = svm.OneClassSVM(nu = 0.01, kernel="poly", gamma= 0.1)
        self.clf3.fit(whites)


        self.clf4 = svm.OneClassSVM(nu = 0.01, kernel="rbf", gamma= 0.1)
        self.clf4.fit(purples)
        print(self.clf4.predict(purples))


        '''
        clf = IsolationForest(contamination= 0.1)
        clf.fit(blacks)
        print(clf.predict(blues))
        '''
        '''
        clf = EllipticEnvelope(store_precision= False, assume_centered=False, contamination=0.1)
        clf.fit(blacks)
        print(clf.predict(blacks))
        '''
        '''
        clf = LocalOutlierFactor(n_neighbors= 20, contamination= 0.1,)
        clf.fit(blacks)
        print(clf.fit_predict(blues))
        '''

        start_time = time.clock()
        #print(c.reshape((-1,3)))
        b = self.clf2.predict(c.reshape((-1,3)))
        b.resize((c.shape[0], c.shape[1]))
        b = 255*(b+1)/2
        #cv2.imshow("b",b)
        #cv2.waitKey(0)
        end_time = time.clock()
        print(end_time - start_time)

        #cv2.imshow("black",b1)#34 35
        #cv2.waitKey(0)

#g = GetPoint()
