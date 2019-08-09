import cv2
import numpy as np
import time
import uvc

def Multicapture(number,interval):
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

    #finds camera and launches window
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("stereoCamera; press space for capture")
    
    count = 0
    imagesL = np.array([[]])
    imagesR = np.array([[]])

    while True:
        ret, frame = cam.read()

        #combines left and right images together
        frame_left = frame.copy()
        frame_left = cv2.flip(frame_left[:, :, 1],0)
        frame_right = frame.copy()
        frame_right = cv2.flip(frame_right[:, :, 2],0)
        
        frame_joint = np.concatenate((frame_left,frame_right), axis=1)

        cv2.imshow("stereoCamera; press space for picture", frame_joint)

        if not ret: 
            break
        k = cv2.waitKey(1)
        
        if k%256 == 32:
            count += 1
            
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        if count > 0 and count <= number:
            #pressing space takes picture and applies calibration
            #print('taking picture and applying calibration...')
            imgL = frame_left
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

            #print('saving rectified images in: input...')
            strcnt = str(count)
            
            globals()['image'+str(count)+'_l_'] = Ldst
            globals()['image'+str(count)+'_r_'] = Rdst

            time.sleep(interval)
            count += 1
            
        if count > number:
            str_number = str(number)
            str_interval = str(interval)
            print(str_number+' pictures taken, at '+str_interval+' intervals')
            print('saved variables in form: image()_r_ or image()_l_')
            count = 0

            print(image1_l_)

            cv2.imwrite('input/test_l_.png',Ldst)
            cv2.imwrite('input/test_r_.png',Rdst)

            
Multicapture(1,0)
