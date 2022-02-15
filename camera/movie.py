import cv2


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
fps = cap.get(cv2.CAP_PROP_FPS)



fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (2560, 960))


while True:
    ret, frame = cap.read()
#    frame = cv2.flip(frame,0)
    cv2.imshow('frame',frame)

    out.write(frame) 
    if cv2.waitKey(1) == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
