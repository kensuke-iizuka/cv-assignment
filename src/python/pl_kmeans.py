import cv2
import numpy as np
import matplotlib as plt
from my_kmeans import *
from numpy import random


cluster_num = 4

input_vector = random.randn(2)

sample_vectors = vectors = random.randn(100, 2)
sample_vectors = sample_vectors.reshape((10,10,2))

label_list = np.zeros((100, 1))
center_list = np.zeros((cluster_num, 2))

# clustering sample vectors
label_list, center_list = k_means(sample_vectors, cluster_num, 3, 100)


input_label = near(input_vector, center_list)

if(input_label == 0): in_color = 'green'
elif(input_label == 1): in_color = 'black'
elif(input_label == 2): in_color = 'red'
else: in_color = 'blue'

plt.plot(input_vector[0],input_vector[1],'o',color=in_color)

sample_vectors = sample_vectors.reshape((100,2))
for vector, label in zip(sample_vectors, label_list):
    if(label == 0): color = 'green'
    elif(label == 1): color = 'black'
    elif(label == 2): color = 'red'
    else: color = 'blue'
    plt.plot(vector[0],vector[1],'x',color=color)

plt.show()
fig = plt.subplots()
fig.savefig("2d_kmeans.png")