import cv2
import numpy as np
import time

Scal = cv2.FileStorage("cal/stereo_calibration.xml", cv2.FILE_STORAGE_READ)
Rot_matrix = Scal.getNode("R").mat()
Tran_dist = Scal.getNode("T").mat()
Scal.release()

#loads in the calibration files and finds the camera matrix and the distortion coefficients
Rcal = cv2.FileStorage("cal/Right_calibration.xml", cv2.FILE_STORAGE_READ)
Rmatrix = Rcal.getNode("cameraMatrix").mat()
Rdist = Rcal.getNode("distCoeffs").mat()
Rcal.release()

Lcal = cv2.FileStorage("cal/left_calibration.xml", cv2.FILE_STORAGE_READ)
Lmatrix = Lcal.getNode("cameraMatrix").mat()
Ldist = Lcal.getNode("distCoeffs").mat()
Lcal.release()

print(Rmatrix, Rdist)
print(Lmatrix, Ldist)

#finds camera and launches window
cam = cv2.VideoCapture(0)
cam.set(15, 10)
cv2.namedWindow("stereoCamera; press space for capture")

t = 1

while True:
    ret, frame = cam.read()

    #combines left and right images together
    frame_left = frame.copy()
    frame_left = cv2.flip(frame_left[:, :, 1],0)
    frame_right = frame.copy()
    frame_right = cv2.flip(frame_right[:, :, 2],0)
    
    frame_joint = np.concatenate((frame_left,frame_right), axis=1)

    cv2.line(frame_joint, (346, 190), (346, 194), (0,0,0), 1)
    cv2.line(frame_joint, (344, 192), (348, 192), (0,0,0), 1)


    cv2.imshow("stereoCamera; press space for picture", frame_joint)

    if not ret: 
        break
    k = cv2.waitKey(1)

    
    if k%256 == 32:
        t += 1
        #pressing space takes picture and applies calibration
        print('taking picture and applying calibration...')
        imgL = frame_left
        strt = str(t)
        cv2.imwrite('input/L_rect_test'+strt+'_l_.png',imgL)
        
        imgR = frame_right
        
        Lheight, Lwidth = imgL.shape[:2]
        Rheight, Rwidth = imgR.shape[:2]

        (leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                Lmatrix, Ldist,
                Rmatrix, Rdist,
                (Lwidth,Lheight), Rot_matrix, Tran_dist,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, -1)

        #undistort
        Lmapx,Lmapy = cv2.initUndistortRectifyMap(Lmatrix,Ldist,leftRectification,leftProjection,(Lwidth,Lheight),5)
        Ldst = cv2.remap(imgL,Lmapx,Lmapy,cv2.INTER_LINEAR)

        Rmapx,Rmapy = cv2.initUndistortRectifyMap(Rmatrix,Rdist,rightRectification,rightProjection,(Rwidth,Rheight),5)
        Rdst = cv2.remap(imgR,Rmapx,Rmapy,cv2.INTER_LINEAR)

        print('saving rectified images in: input...')
        
        cv2.imwrite('input/R_rect'+strt+'_r_.png',Rdst)
        cv2.imwrite('input/L_rect'+strt+'_l_.png',Ldst)
        time.sleep(0.1)
        
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
    
