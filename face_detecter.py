import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src = cv.imread("0001.jpg")

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
 
# Detect people in the image
(rects, weights) = hog.detectMultiScale(src,winStride=(5,5),padding=(4,4),scale=1.4,
useMeanshiftGrouping=False)
for (x, y, w, h) in rects:
    cv.rectangle(src, (x, y), (x + w, y + h), (200, 255, 0), 3)

plt.imshow(src[:,:,::-1])

