import cv2
from cv2 import CV_8S
import numpy as np
import math
import os
import random
import csv
# imgR = cv2.imread("/home/bj/data/dnn/cfnet_venv/CFNet/datasets/kitti12_15/Kitti/testing/image_3/000010_10.png",cv2.IMREAD_COLOR)
# imgL = cv2.imread("/home/bj/data/dnn/cfnet_venv/CFNet/datasets/kitti12_15/Kitti/testing/image_2/000010_10.png",cv2.IMREAD_COLOR)
img = cv2.imread("/home/bj/data/dnn/CFNet/camera/movie120.png", cv2.IMREAD_COLOR)


img = cv2.resize(img, dsize=(1280,960))
#cv2.imshow('img',imgL)
max = np.max(img)
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
 

#img = 1/img
#mask = img > 0.051
#img[mask] = 0.051
#cimg[:,:,:] = img[:,:,:] * 1

cimg = cv2.cvtColor(cimg,cv2.COLOR_HLS2BGR)
# cv2.imshow('Right image',imgR)
# cv2.imshow('left image',imgL)
cv2.imshow('cimg',cimg)

cv2.waitKey()
cv2.destroyAllWindows()