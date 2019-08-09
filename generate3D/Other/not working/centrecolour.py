import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageGrab
import cv2

  
filename = "output/disparity/test_disparity3.png"
with Image.open(filename) as image: 
    width, height = image.size

half_height = int(height / 2)
half_width = int(width / 2)

print(half_height, half_width)
print(type(half_width))

img = cv2.imread('output/disparity/test_disparity3.png')

print(img[half_height,half_width])

Sum = 0
count = 0

for i in (-2, -1, 0, 1, 2):
    for j in (-2, -1, 0, 1, 2):
        Sum = Sum + img[half_height + i, half_width + j][0]
        print(img[half_height + i, half_width + j][0])
        #print(Sum)
        count += 1

average = Sum / count
print(average)
