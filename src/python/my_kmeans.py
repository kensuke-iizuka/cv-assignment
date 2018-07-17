import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#Define const
CLUSTER_NUM = 4
# ベクトルを構成する要素数 画像ならRGBで３次元かな
ELEMENT_NUM = 2 
# ベクトルのかず
VECTOR_NUM = 100

def euclid_dist(vector1, vector2):
    """
    vector1とvector2のユークリッド距離を返す
    :param vector1:
    :param vector2:
    :return:
    """
    sum = 0
    for i in range(vector1.size):
        sum += (vector1[i] - vector2[i]) ** 2
    
    dist = sum ** 0.5
    return dist

def near(vector, center_vectors):
    """
    vectorに対し、尤も近いラベルを返す
    :param vector:
    :param center_vectors(dimension, cluster_num):
    :return:
    """
    dist_list = np.zeros(center_vectors.shape[0])
    i = 0
    for i, center_vector in enumerate(center_vectors):
        dist_list[i] = euclid_dist(vector, center_vector)

    return np.argmin(dist_list) 


def clustering(vectors, label_count=CLUSTER_NUM, learning_count_max=1000):
    """
    K-meansを行い、各ラベルの重心を返す
    :param vectors:
    :param label_count:
    :param learning_count_max:
    :return:
    """
    label_vector = random.randint(0, high=label_count, size=VECTOR_NUM)
    #一つ前のStepで割り当てられたラベル。終了条件の判定に使用
    old_label_vector = np.array(VECTOR_NUM) 
    #各クラスタの重心vector
    # center_vectors = [[0 for i in range(len(vectors[0]))] for label in range(label_count)]
    center_vectors = np.zeros((CLUSTER_NUM, ELEMENT_NUM))

    for step in range(learning_count_max):
        #各クラスタの重心vectorの作成
        for vec, label in zip(vectors, label_vector):
            center_vectors[label] = [c+v for c, v in zip(center_vectors[label], vec)]
        for i, center_vector in enumerate(center_vectors):
            center_vectors[i] = [v/len(np.where(label_vector==i)[0]) for v in center_vector]
        #各ベクトルのラベルの再割当て
        for i, vec in enumerate(vectors):
            label_vector[i] = near(vec, center_vectors)

        #前Stepと比較し、ラベルの割り当てに変化が無かったら終了
        print(type(old_label_vector))
        print(type(label_vector))
        if (old_label_vector == label_vector).any():
            break
        #ラベルのベクトルを保持
        old_label_vector = [l for l in label_vector]
    return center_vectors


input_vector = random.randn(2)

sample_vectors = vectors = random.randn(VECTOR_NUM, ELEMENT_NUM)

center_vectors = clustering(sample_vectors)

input_label = near(input_vector, center_vectors)

if(input_label == 0): in_color = 'green'
elif(input_label == 1): in_color = 'yellow'
elif(input_label == 2): in_color = 'red'
else: in_color = 'blue'

plt.plot(input_vector[0],input_vector[1],'o',color=in_color)


for vector in sample_vectors:
    label = near(vector,center_vectors)
    if(label == 0): color = 'green'
    elif(label == 1): color = 'yellow'
    elif(label == 2): color = 'red'
    else: color = 'blue'
    plt.plot(vector[0],vector[1],'x',color=color)


plt.show()