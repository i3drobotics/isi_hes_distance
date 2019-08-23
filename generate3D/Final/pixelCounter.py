import cv2
import numpy as np

def counter():
    pixelCount = pixelCounter()
    return pixelCount

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

def initVideoCamera():
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

pixelCount = False
while pixelCount is False:
    pixelCount = counter()
    print(pixelCount)
 
