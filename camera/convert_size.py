import numpy as np
import cv2

name = '/home/bj/data/dnn/CFNet/etc/im0.png'
img = cv2.imread(name)
img = cv2.resize(img, dsize=(1242, 375), interpolation=cv2.INTER_AREA)
cv2.imwrite('/home/bj/data/dnn/CFNet/etc/im0f.png',img)