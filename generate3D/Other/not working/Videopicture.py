import cv2
import numpy as np
import os
from PIL import Image

# Playing video from file:
def frames(file):
    cap = cv2.VideoCapture(file)

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(currentFrame < 20):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in png file
        name = './data/frame' + str(currentFrame) + '.png'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def cutter(file, number):
    img = Image.open(file)
    width,height = img.size

    half_width = int(width / 2)
    
    crop_img1 = img.crop( ( half_width, 0, width, height ) )
    crop_img2 = img.crop( ( 0, 0, half_width, height ) )
    crop_img1.save('input/'+number+'crop_img_l_.png')
    crop_img2.save('input/'+number+'crop_img_r_.png')

frames('stereo_video_20190704_171103_011.mp4')

for i in range(20):
    j = str(i)
    cutter('data/frame'+j+'.png',j)
    print(i+1,'is done')
