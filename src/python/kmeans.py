import numpy as np
import cv2 as cv
from klib import *

img = cv.imread('../img/metro.jpg')
dist_img_path = "../img/metro_opencv.jpg"

# src_img = read_image(img_path)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
save_image(res2, dist_img_path)
# cv.imshow('res2',res2)
# cv.waitKey(0)
# cv.destroyAllWindows()
