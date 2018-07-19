
import cv2
import numpy as np
import matplotlib as plt
from klib import *
from gaussian_mixture import *

img_path = "../img/umi2.jpg"
dist_img_path = "../img/gaussian_result.jpg"

src_img = read_image(img_path)
dist_img = src_img

img_height = src_img.shape[0]
img_width = src_img.shape[1]

print("img_height:{0}, img_width:{1}".format(src_img.shape[0], src_img.shape[1]))


model = GaussianMixture(3)

input_vectors = src_img.reshape(img_height * img_width, 3)
model.fit(input_vectors, iter_max=5)
labels = model.classify(input_vectors)

# print(labels.shape)
labels = labels.reshape(img_height, img_width)

for i in range(img_height):
    for j in range(img_width):
        label = labels[i][j]
        if(label == 0): 
            dist_img[i][j][0] = 0 # B
            dist_img[i][j][1] = 0 # G
            dist_img[i][j][2] = 255 # R

        elif(label == 1): 
            dist_img[i][j][0] = 0
            dist_img[i][j][1] = 255
            dist_img[i][j][2] = 0

        elif(label == 2): 
            dist_img[i][j][0] = 255
            dist_img[i][j][1] = 0
            dist_img[i][j][2] = 0

        else: 
            dist_img[i][j][0] = 115
            dist_img[i][j][1] = 115
            dist_img[i][j][2] = 155


# save_image(dist_img, dist_img_path)
# show_image(dist_img)