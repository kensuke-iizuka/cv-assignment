import cv2
import numpy as np
import matplotlib as plt
from klib import *
from my_kmeans import k_means

img_path = "../img/umi.jpg"
dist_img_path = "../img/kmeans_result.jpg"

src_img = read_image(img_path)
dist_img = src_img

img_height = src_img.shape[0]
img_width = src_img.shape[1]

print("img_height:{0}, img_width:{1}".format(src_img.shape[0], src_img.shape[1]))

label_list = np.zeros((img_height*img_width))

input_vectors = src_img.reshape(img_height * img_width, 3)
label_list = k_means(input_vectors, 4, 3, 1000)

label1 = len(np.where(label_list==0)[0])
label2 = len(np.where(label_list==1)[0])
label3 = len(np.where(label_list==2)[0])
label4 = len(np.where(label_list==3)[0])

re_label_list = label_list.reshape(img_height, img_width)



for i in range(img_height):
    for j in range(img_width):
        label = re_label_list[i][j]
        if(label == 0): 
            dist_img[i][j][0] += src_img[i][j][0] / label1 # B
            dist_img[i][j][1] += src_img[i][j][1] / label1 # G
            dist_img[i][j][2] += src_img[i][j][2] / label1 # R

        elif(label == 1): 
            dist_img[i][j][0] += src_img[i][j][0] / label2
            dist_img[i][j][1] += src_img[i][j][1] / label2
            dist_img[i][j][2] += src_img[i][j][2] / label2

        elif(label == 2): 
            dist_img[i][j][0] += src_img[i][j][0] / label3
            dist_img[i][j][1] += src_img[i][j][1] / label3
            dist_img[i][j][2] += src_img[i][j][2] / label3

        else: 
            dist_img[i][j][0] += src_img[i][j][0] / label4
            dist_img[i][j][1] += src_img[i][j][1] / label4
            dist_img[i][j][2] += src_img[i][j][2] / label4


save_image(dist_img, dist_img_path)
# show_image(dist_img)
