import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#Define const
CLUSTER_NUM = 4
# ベクトルを構成する要素数 画像ならRGBで３次元かな
ELEMENT_NUM = 3 
# ベクトルのかず
VECTOR_NUM = 100

def euclid_dist(vector1, vector2):
    """
    vector1とvector2のユークリッド距離を返す
    :param vector1: size = dim
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
    :param vector: size = dim
    :param center_vectors(size = (dimension, cluster_num)):
    :return:
    """
    dist_list = np.zeros(center_vectors.shape[0])
    i = 0
    for i, center_vector in enumerate(center_vectors):
        dist_list[i] = euclid_dist(vector, center_vector)

    return np.argmin(dist_list) 


def clustering(vectors, cluster_num, max_itr=1000):
    """
    K-meansを行い、各ラベルの重心を返す
    :param vectors: size=(height*width, dim)
    :return: center_vectors
    """
    label_vector = random.randint(0, high=cluster_num, size=vectors.shape[0])
    # print(label_vector.shape)
    #一つ前のStepで割り当てられたラベル。終了条件の判定に使用
    old_label_vector = np.array(vectors.shape[0]) 
    #各クラスタの重心vector
    # center_vectors = [[0 for i in range(len(vectors[0]))] for label in range(label_count)]
    center_vectors = np.zeros((cluster_num, vectors.shape[1]))

    for step in range(max_itr):
        #各クラスタの重心vectorの作成
        for vec, label in zip(vectors, label_vector):
            center_vectors[label] = [c+v for c, v in zip(center_vectors[label], vec)]

        for i, center_vector in enumerate(center_vectors):
            center_vectors[i] = [v/len(np.where(label_vector==i)[0]) for v in center_vector]
        #各ベクトルのラベルの再割当て
        for i, vec in enumerate(vectors):
            label_vector[i] = near(vec, center_vectors)

        #前Stepと比較し、ラベルの割り当てに変化が無かったら終了
        if (old_label_vector == label_vector).any():
            break
        #ラベルのベクトルを保持
        old_label_vector = [l for l in label_vector]
    return center_vectors

def k_means(vectors, cluster_num, dim, itr_num):
    """
    :param vectors: input vectors ex)input img(size = (height, width, dim))
    :param cluster_num: how many category you want to divide space
    :param itr_num: max number of update centroid
    :return label_vectors,center_vectors: return label of each input vector
    """

    label_vectors = np.zeros(vectors.shape[0] * vectors.shape[1])
    vectors = vectors.reshape((vectors.shape[0] * vectors.shape[1], vectors.shape[2]))
    print(vectors.shape)

    center_vectors = clustering(vectors, cluster_num, itr_num)
    
    for i, vector in enumerate(vectors):
        label_vectors[i] = near(vector, center_vectors)
    
    label_vectors = np.uint8(label_vectors.reshape((label_vectors.shape[0], 1)))
    
    return label_vectors, center_vectors
