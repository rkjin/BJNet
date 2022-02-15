import cv2
import numpy as np
import math
import os
import random
import csv
imgR = cv2.imread("/home/bj/data/dnn/CFNet/etc/im1f.png",cv2.IMREAD_COLOR)
img = cv2.imread("/home/bj/data/dnn/CFNet/etc/im0f.png", cv2.IMREAD_COLOR)
cv2.imshow('img',imgR)
max = np.max(img)
print(max)
cimg = np.zeros_like(img)
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

cimg[:,:,:] = img[:,:,:] * (255 / max) 
cimg = cv2.cvtColor(cimg,cv2.COLOR_HLS2BGR)
cv2.imshow('Right image',imgR)
cv2.imshow('cimg',cimg)

cv2.waitKey()
cv2.destroyAllWindows()