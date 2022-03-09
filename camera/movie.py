import cv2


cap = cv2.VideoCapture('/home/bj/data/dnn/BJNet/camera/output4.avi')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# fps = cap.get(cv2.CAP_PROP_FPS)


ret, frame = cap.read()
print(frame.shape)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc,  3, (frame.shape[1],frame.shape[0]))


while True:
    ret, frame = cap.read()
#    frame = cv2.flip(frame,0)
    cv2.imshow('frame',frame)

    out.write(frame) 
    if cv2.waitKey(300) == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
