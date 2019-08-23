import cv2
import numpy as np

def checker():
    average,error = sanityCheck()
    return average,error
    
def sanityCheck():
    cam = initVideoCamera2()

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

def initVideoCamera2():
    #finds Deimos camera
    cam = cv2.VideoCapture(0)
    return cam

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
   
