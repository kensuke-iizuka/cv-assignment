import numpy as numpy
import cv2

img = cv2.imread('../img/umi.jpg')
# img = cv2.GaussianBlur(img, (11, 11), 0)
img_shifted = cv2.pyrMeanShiftFiltering(img, 21, 21)
cv2.imwrite("../img/mean_shift.jpg", img_shifted)