import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import argparse
import signal
import math
import os
import pptk





def ranger():
    distance = getRange()
    return distance

def checker():
    average,error = sanityCheck()
    output = (average,error)
    return output

def counter():
    pixelCount = pixelCounter()
    return pixelCount





def getRange():
    dist = -1

    maxImageCount = 35
    timeBetweenPics = 0.03
    totalDist = 0
    
    Rot_matrix,Tran_dist,Q_matrix,Rmatrix,Rdist,Lmatrix,Ldist = loadCalibration()
    cam = initVideoCamera()
    

    for imageNum in range(maxImageCount):
        frame_right,frame_left = getStereoImages(cam)
        rectR,rectL = rectifyStereoImages(frame_right,frame_left,Rot_matrix,Tran_dist,Rmatrix,Rdist,Lmatrix,Ldist)
        points = generate3D(rectR,rectL,Q_matrix)
        origXPos,origYPos,origZPos = pointsToCoordinates(points)
        adjustXPos,adjustYPos = coordinateAdjustment(origXPos,origYPos)
        distToPoint = distToClosestPointsToOrigin(adjustXPos,adjustYPos,origZPos)
                
        totalDist += distToPoint
        time.sleep(timeBetweenPics)

    dist = averageDistance(totalDist,maxImageCount)
        
    return dist

def loadCalibration():
    #loads in the stereo calibration
    script_path = os.path.dirname(os.path.realpath(__file__))
    Scal = cv2.FileStorage(script_path+"/cal/stereo_calibration.xml", cv2.FILE_STORAGE_READ)
    Rot_matrix = Scal.getNode("R").mat()
    Tran_dist = Scal.getNode("T").mat()
    Q_matrix = Scal.getNode("Q").mat()
    Scal.release()

    #loads in the calibration files and finds the camera matrix and the distortion coefficients
    Rcal = cv2.FileStorage(script_path+"cal/Right_calibration.xml", cv2.FILE_STORAGE_READ)
    Rmatrix = Rcal.getNode("cameraMatrix").mat()
    Rdist = Rcal.getNode("distCoeffs").mat()
    Rcal.release()

    Lcal = cv2.FileStorage(script_path+"cal/left_calibration.xml", cv2.FILE_STORAGE_READ)
    Lmatrix = Lcal.getNode("cameraMatrix").mat()
    Ldist = Lcal.getNode("distCoeffs").mat()
    Lcal.release()

    return Rot_matrix,Tran_dist,Q_matrix,Rmatrix,Rdist,Lmatrix,Ldist

def initVideoCamera():
    cam = cv2.VideoCapture(0)
    return cam

def getStereoImages(cam):
    ret, frame = cam.read()
    frame_left = frame.copy()
    frame_left = cv2.flip(frame_left[:, :, 1],0)
    frame_right = frame.copy()
    frame_right = cv2.flip(frame_right[:, :, 2],0)

    return frame_right,frame_left 

def rectifyStereoImages(frame_right,frame_left,Rot_matrix,Tran_dist,Rmatrix,Rdist,Lmatrix,Ldist):
    #pressing space takes picture and applies calibration
    
    Lheight, Lwidth = frame_left.shape[:2]
    Rheight, Rwidth = frame_right.shape[:2]

    (leftRectification, rightRectification, leftProjection, rightProjection,
    dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
            Lmatrix, Ldist,
            Rmatrix, Rdist,
            (Lwidth,Lheight), Rot_matrix, Tran_dist,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, -1)

    #undistort
    Lmapx,Lmapy = cv2.initUndistortRectifyMap(Lmatrix,Ldist,leftRectification,leftProjection,(Lwidth,Lheight),5)
    Ldst = cv2.remap(frame_left,Lmapx,Lmapy,cv2.INTER_LINEAR)

    Rmapx,Rmapy = cv2.initUndistortRectifyMap(Rmatrix,Rdist,rightRectification,rightProjection,(Rwidth,Rheight),5)
    Rdst = cv2.remap(frame_right,Rmapx,Rmapy,cv2.INTER_LINEAR)

    return Rdst,Ldst

def generate3D(rectR,rectL,Q_matrix):
    CV_MATCHER_BM = 1
    CV_MATCHER_SGBM = 0

    #define matching parameters
    #SETTINGS FOR PHOBOS NUCLEAR  
    algorithm = CV_MATCHER_BM
    window_size = 21
    block_size = 15
    min_disp = -69
    num_disp = 16*10
    uniqness_ratio = 15
    speckle_window_size = 500
    speckle_range = 5
    
    #Convert source image to unsigned 8 bit integer Numpy array
    arrL = np.uint8(rectL)
    arrR = np.uint8(rectR)

    #generate disparity using stereo matching algorithms
    if algorithm == CV_MATCHER_BM:
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
        stereo.setMinDisparity(min_disp)
        stereo.setSpeckleWindowSize(speckle_window_size)
        stereo.setSpeckleRange(speckle_range)
        stereo.setUniquenessRatio(uniqness_ratio)
    elif algorithm == CV_MATCHER_SGBM:
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                blockSize = block_size,
                P1 = 8*3*window_size**2,
                P2 = 32*3*window_size**2,
                disp12MaxDiff = 1,
                uniquenessRatio = uniqness_ratio,
                speckleWindowSize = speckle_window_size,
                speckleRange = speckle_range
                )
    
    disp = stereo.compute(arrL, arrR).astype(np.float32) / 16.0

    #reproject disparity to 3D
    points = cv2.reprojectImageTo3D(disp, Q_matrix)
    disp = (disp-min_disp)/num_disp

    #normalise disparity
    imask = disp > disp.min()
    disp_thresh = np.zeros_like(disp, np.uint8)
    disp_thresh[imask] = disp[imask]

    disp_norm = np.zeros_like(disp, np.uint8)
    cv2.normalize(disp, disp_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #format colour image from left camera for mapping to point cloud
    h, w = arrL.shape[:2]
    colors = cv2.cvtColor(arrL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    #saves the points of a point map as variables
    return out_points

def pointsToCoordinates(points):
    #takes list of all points and breaks it to 3 coordinate lists
    originalXPositions = []
    originalYPositions = []
    originalZPositions = []

    for coordinate in range(len(points)):
        originalXPositions.append(points[coordinate][0])
        originalYPositions.append(points[coordinate][1])
        originalZPositions.append(points[coordinate][2])

    return originalXPositions,originalYPositions,originalZPositions

def coordinateAdjustment(originalXPositions,originalYPositions):
    #Original origin is with respect to the camera, adjusted origin is with respect to the laser spot at 1.0m
    changeInX = 0.040
    changeInY = 0.066

    adjustedXPositions = []
    adjustedYPositions = []

    for xValue in originalXPositions:
        adjustedXValue = xValue + changeInX
        adjustedXPositions.append(adjustedXValue)

    for yValue in originalYPositions:
        adjustedYValue = yValue + changeInY
        adjustedYPositions.append(adjustedYValue)

    return adjustedXPositions,adjustedYPositions

def distToClosestPointsToOrigin(adjustedXPositions,adjustedYPositions,originalZPositions):
    #Uses pythagoras to find the shortest disance to the adjusted origin, and then
    #the position in the list of this, and therefore the z coordinate of this point
    listPoints = []
    shortestDist = 10.0

    for coord in range(len(adjustedXPositions)):
        distFromOrigin = math.sqrt(adjustedXPositions[coord]*adjustedXPositions[coord] + adjustedYPositions[coord]*adjustedYPositions[coord])
        if distFromOrigin < shortestDist:
            shortestDist = distFromOrigin
            listPoints.append(coord)

    distToPoint = originalZPositions[listPoints[-1]]
    
    return distToPoint

def averageDistance(totalDist,maxImageCount):
    averageDistance = totalDist / float(maxImageCount)
    #subtracting difference from the front of the Deismos camera and the probe front
    averageDistance -= 0.03
    averageDistance = round((averageDistance),3)
    
    return averageDistance





def sanityCheck():
    cam = initVideoCamera()

    totalDistance = 0
    maxImageCount = 10
    
    for num in range(maxImageCount):
        imageLeft = getStereoImage(cam)
        threshedLeft = thresholder(imageLeft)

        blurredLeft = blur(threshedLeft)
        xPixel,yPixel = maxBrightness(blurredLeft)
        yDistance = calibrationY(yPixel)
        totalDistance += yDistance

    average = averageDistance2(totalDistance,maxImageCount)
    average = rounder(average)

    if average > 2 or average < 0.3:
        error = True
    else:
        error = False
        
    return average,error

def getStereoImage(cam):
    #produces an image from the left camera 
    ret, frame = cam.read()
    frame_left = frame.copy()
    frame_left = cv2.flip(frame_left[:, :, 1],0)

    frame_left = frame_left[0:216, 330:441]

    return frame_left

def thresholder(image):
    #Applies a binary threshold which takes any pixels that over expose the camera
    ret,thresh = cv2.threshold(image,254,255,cv2.THRESH_BINARY)
    return thresh

def blur(thresh):
    #opening and bluring the image
    blurred = cv2.GaussianBlur(thresh, (1,1), 0)
    return blurred

def maxBrightness(blurred):
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
    xpix = maxLoc[0] + 329
    ypix = maxLoc[1]
    return xpix,ypix

def calibrationY(ypix):
    yDistance = 0.0000000276*(ypix)**4 - 0.0000151683*(ypix)**3 + 0.0031651809*(ypix)**2 - 0.2914654102*(ypix)**1 + 10.3118973406
    return yDistance

def averageDistance2(totalDistance,maxImageCount):
    average = totalDistance / float(maxImageCount)
    return average

def rounder(yDistance, base=0.05):
    yDistance =  base * round(yDistance/base)
    yDistance = round(yDistance, 2)
    return yDistance





def pixelCounter():
    
    cam = initVideoCamera()
    imageLeft = getStereoImage(cam)
    threshedLeft = thresholder(imageLeft)
    
    count = 0
    for row in range(len(threshedLeft)):
        for column in range(len(threshedLeft[0])):
            if threshedLeft[row][column] == 255:
                count += 1
                
    if count <= 30:
        pixelCount = True
    else:
        pixelCount = False
            
    return pixelCount



