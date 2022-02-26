import cv2
from cv2 import CV_8S
import numpy as np
import math
import os
import random
import csv
# imgR = cv2.imread("/home/bj/data/dnn/cfnet_venv/CFNet/datasets/kitti12_15/Kitti/testing/image_3/000010_10.png",cv2.IMREAD_COLOR)
# imgL = cv2.imread("/home/bj/data/dnn/cfnet_venv/CFNet/datasets/kitti12_15/Kitti/testing/image_2/000010_10.png",cv2.IMREAD_COLOR)
img = cv2.imread("/home/bj/data/dnn/BJNet/camera/000000_102.png", cv2.IMREAD_COLOR)
 
img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

img = cv2.applyColorMap(img , cv2.COLORMAP_JET)

cv2.imshow('img',img)

cv2.waitKey()
cv2.destroyAllWindows()