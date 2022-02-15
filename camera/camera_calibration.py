import numpy as np
import cv2
import glob
cap = cv2.VideoCapture(0)
# termination criteria
# cv2.TERM_CRITERIA_EPS - 정해둔 정확도에 다다르면 알고리즘 반복을 멈춘다
# cv2.TERM_CRITERIA_MAX_ITER - 지정한 반복 수만큼을 지나면 알고리즘을 멈춘다.
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - 위의 조건 중 하나라도 만족하면 알고리즘을 멈춘다.
# criteria = (a,b,c) 형태로 쓴다면, a가 위의 방식이고 b는 iterations 수, c는 요구되는 정확도(epsilon)이다!
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
print(objp.shape)  # (54, 3)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
print((np.mgrid[0:9,0:6]).shape) # (2,9,6)
print(np.mgrid[0:9,0:6].T) # (6,9,2)
print(objp.shape) # (54,3)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob('./*.jpg')
#for fname in images:
for _ in range(10):
    ret, img = cap.read()
#    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
    cv2.imshow('img',img)
    cv2.waitKey(500)
cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#print(ret, mtx, dist, rvecs, tvecs)
img = cv2.imread('/home/bj/data/dnn/CFNet/etc/chessboard.png')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
print("total error: ", tot_error/len(objpoints))
#저장
np.savez('./calib.npz',ret=ret,mtx=mtx,dist=dist,rvecs=rvecs,tvecs=tvecs)
camera = np.load('./calib.npz')
print(camera['ret'],camera['mtx'],camera['dist'],camera['rvecs'],camera['tvecs'])