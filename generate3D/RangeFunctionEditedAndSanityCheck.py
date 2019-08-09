import cv2
import numpy as np
import time
import signal
import math
import os

def ranger():
    dist = getRange()
    return dist

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
    Scal = cv2.FileStorage("cal/stereo_calibration.xml", cv2.FILE_STORAGE_READ)
    Rot_matrix = Scal.getNode("R").mat()
    Tran_dist = Scal.getNode("T").mat()
    Q_matrix = Scal.getNode("Q").mat()
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
    window_size = 30
    block_size = 15
    min_disp = 20
    num_disp = 16*6
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
    averageDistance = round((averageDistance),3)
    
    return averageDistance




def sanityCheck():
    cam = cv2.VideoCapture(0)

    def Lo_xpixdist(xpix):
        return 1.66641476680862*10**-7*(xpix)**4 - 0.00025723591349712*(xpix)**3 + 0.148802849361544*(xpix)**2 + -38.2381266165862*(xpix) + 3684.10972750337

    def Hi_xpixdist(xpix):
        return (-2.0 / 515.0)*(xpix) + (945.0 / 515.0)

    def Hi_ypixdist(ypix):
        return 6.16848544623133*10**-9*(ypix)**4-2.32030083903941*10**-6*(ypix)**3 + 0.000300058782171953*(ypix)**2 - 0.0118573916872868*(ypix)**1 + 0.27789595721237

    def Lo_ypixdist(ypix):
        return (1.0 / 660.0)*(ypix) + (31.0 / 165.0)


    ret, frame = cam.read()

    #combines left and right images together
    frame_left = frame.copy()
    frame_left = cv2.flip(frame_left[:, :, 1],0)
    frame_right = frame.copy()
    frame_right = cv2.flip(frame_right[:, :, 2],0)

    frame_joint = np.concatenate((frame_left,frame_right), axis=1)

    cv2.line(frame_joint, (346, 188), (346, 192), (0,0,0), 1)
    cv2.line(frame_joint, (344, 190), (348, 190), (0,0,0), 1)

    image_l_ = frame_left

    gray = image_l_[0:281, 260:441]

    #name a greyscale image to be analysed for its brightest region
    #finding this, use the location to pride a sanity check for distance 
    gray = image_l_[0:281, 260:441]

    #apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (71,71), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray) 
    image = gray.copy()
    cv2.circle(image, maxLoc, 7, (0, 0, 0), 1)
    cv2.circle(image, maxLoc, 9, (255, 0, 0), 1)

    #find the distance due to the location of the brightest region
    xpix = maxLoc[0] + 259
    ypix = maxLoc[1]

    #Applying the two possible funcions to each of the x and y pixel location depeding on where the pixel is
    #two lines used to fit the best to the data for moving the the laser spot
    if 394 <= xpix:
        x_dist = Hi_xpixdist(xpix)
    else:
        x_dist = Lo_xpixdist(xpix)

    if 74 >= ypix:
        y_dist = Lo_ypixdist(ypix)
    else:
        y_dist = Hi_ypixdist(ypix)

    #print()
    #print(xpix, x_dist)
    #print(ypix, y_dist)
    #print()

    av_pix_dist = (x_dist + y_dist) / 2

    #if the spot is in an unrecognised location on the camera, produces an error otherwise prints distance
    if abs(av_pix_dist - x_dist) > 0.25 or abs(av_pix_dist - y_dist) > 0.25:
        Dist = 0
        Error = 'Error'

    else:
        av_pix_dist = round(av_pix_dist,2)

        Dist = av_pix_dist
        Error = 'No Error'

    Tup = (Dist, Error)
    return Tup
