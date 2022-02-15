import cv2
from cv2 import VideoCapture


# cap = cv2.VideoCapture('/home/bj/data/dnn/CFNet/output2.avi')
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# fps = cap.get(cv2.CAP_PROP_FPS)
cap = VideoCapture(5)
while True:
    ret, frame = cap.read()
  
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
