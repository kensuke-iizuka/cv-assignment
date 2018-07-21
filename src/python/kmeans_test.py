import cv2
import numpy as np
import matplotlib as plt
from klib import *
from my_kmeans import k_means

img_path = "../img/metro.jpg"
dist_img_path = "../img/metro_result.jpg"

src_img = read_image(img_path)

img_height = src_img.shape[0]
img_width = src_img.shape[1]
img_dim = src_img.shape[2]

dist_img = np.zeros((img_height, img_width, img_dim))

cluster_num = 8

print("img_height:{0}, img_width:{1}".format(src_img.shape[0], src_img.shape[1]))

label_list = np.zeros((img_height * img_width, 1))
center_list = np.zeros((cluster_num, img_dim))

label_list, center_list = k_means(src_img, cluster_num, 3, 100)

center_list = np.uint8(center_list)
res = center_list[label_list.flatten()]
dist_img = res.reshape((src_img.shape))

save_image(dist_img, dist_img_path)

# label1 = len(np.where(label_list==0)[0])
# label2 = len(np.where(label_list==1)[0])
# label3 = len(np.where(label_list==2)[0])
# label4 = len(np.where(label_list==3)[0])

# for i in range(img_height):
#     for j in range(img_width):
#         label = re_label_list[i][j]
#         if(label == 0): 
#             dist_img[i][j][0] += src_img[i][j][0] / label1 # B
#             dist_img[i][j][1] += src_img[i][j][1] / label1 # G
#             dist_img[i][j][2] += src_img[i][j][2] / label1 # R

#         elif(label == 1): 
#             dist_img[i][j][0] += src_img[i][j][0] / label2
#             dist_img[i][j][1] += src_img[i][j][1] / label2
#             dist_img[i][j][2] += src_img[i][j][2] / label2

#         elif(label == 2): 
#             dist_img[i][j][0] += src_img[i][j][0] / label3
#             dist_img[i][j][1] += src_img[i][j][1] / label3
#             dist_img[i][j][2] += src_img[i][j][2] / label3

#         else: 
#             dist_img[i][j][0] += src_img[i][j][0] / label4
#             dist_img[i][j][1] += src_img[i][j][1] / label4
#             dist_img[i][j][2] += src_img[i][j][2] / label4


# show_image(dist_img)
