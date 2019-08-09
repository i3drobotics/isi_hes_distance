import cv2
import numpy as np

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


