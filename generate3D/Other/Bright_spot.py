import cv2
import numpy as np
import time
import argparse

def xpixdist(xpix):
    return 1.66641476680862*10**-7*(xpix)**4 - 0.00025723591349712*(xpix)**3 + 0.148802849361544*(xpix)**2 + -38.2381266165862*(xpix) + 3684.10972750337

def ypixdist(ypix):
    return 6.16848544623133*10**-9*(ypix)**4-2.32030083903941*10**-6*(ypix)**3 + 0.000300058782171953*(ypix)**2 - 0.0118573916872868*(ypix)**1 + 0.27789595721237

def Multicapture(number,interval):
    Scal = cv2.FileStorage("cal/Test/stereo_calibration.xml", cv2.FILE_STORAGE_READ)
    Rot_matrix = Scal.getNode("R").mat()
    Tran_dist = Scal.getNode("T").mat()
    Scal.release()

    #loads in the calibration files and finds the camera matrix and the distortion coefficients
    Rcal = cv2.FileStorage("cal/Test/Right_calibration.xml", cv2.FILE_STORAGE_READ)
    Rmatrix = Rcal.getNode("cameraMatrix").mat()
    Rdist = Rcal.getNode("distCoeffs").mat()
    Rcal.release()

    Lcal = cv2.FileStorage("cal/Test/left_calibration.xml", cv2.FILE_STORAGE_READ)
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
            #saves the rectified left and right pictures
            str_number = str(number)
            str_interval = str(interval)
            print(str_number+' pictures taken, at '+str_interval+' intervals')
            count = 0

            gray = image1_l_

            # apply a Gaussian blur to the image then find the brightest region
            
            gray = cv2.GaussianBlur(gray, (7,7), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray) 
            image = gray.copy()
            cv2.circle(image, maxLoc, 7, (0, 0, 0), 1)
            cv2.circle(image, maxLoc, 9, (255, 0, 0), 1)

            # display the results of our newly improved method
            cv2.imshow("Brightest Area", image)
            xpix = maxLoc[0]
            ypix = maxLoc[1]
            print()
            print(xpix, xpixdist(xpix))
            print(ypix, ypixdist(ypix))
            print()
            av_pix_dist = (xpixdist(xpix) + ypixdist(ypix)) / 2
            if abs(av_pix_dist - xpixdist(xpix)) > 0.05 or abs(av_pix_dist - ypixdist(ypix)) > 0.05:
                print("Distance can't be determined, likely unaligned")
            else:
                av_pix_dist = round(av_pix_dist,2)
                print('Distance to object is near:', av_pix_dist,'m')
        
Multicapture(1,0.0)

"""

Post Lunch plan:

Move target to different places and measure the location of the centre of the laser spot
if there is a line find the line equation
compare the position of the brightest spot with the position it expects to be
maybe take measurements of a few of the pictures, see if they differ a lot

find a way to find the distance to a point 


"""

















