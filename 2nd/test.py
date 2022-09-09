import cv2

# cap = cv2.VideoCapture('/home/bj/data/dnn/CFNet/output2.avi')
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# fps = cap.get(cv2.CAP_PROP_FPS)
capl = cv2.VideoCapture('/home/bj/data/dnn/BJNet/ioutputl.avi')
capr = cv2.VideoCapture('/home/bj/data/dnn/BJNet/ioutputr.avi')
n = 0
while True:
    n += 1
    ret, framel = capl.read()
    ret, framer = capl.read()
    framel = cv2.resize(framel, dsize=(1315,375),interpolation=cv2.INTER_AREA)
    cv2.imshow('framel',framel[:,:])
    cv2.imshow('framer',framer[:,:])
    if n == 35:
        input()
    if cv2.waitKey(1) == 27:
        break

capl.release()
capr.release()
cv2.destroyAllWindows()
