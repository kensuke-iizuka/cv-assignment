import pandas as pd
import numpy as np
from numpy import random

def create_data(size, x_mean, x_stddiv, y_mean, y_stddiv):
    data = random.randn(size, 2)
    for i in data:
        i[0] = i[0] * x_stddiv + x_mean
        i[1] = i[1]  * y_stddiv + y_mean
    return data
     
cluster1 = create_data(100, 2, 0.5, 2, 1)
cluster2 = create_data(100, -2, 0.5, 0, 0.5)
cluster3 = create_data(100, 2, 1, -2, 0.5)

def k_means(n_cluster, data, iteration=100):
    # 初期値
    centroid = np.zeros([3, 2])
    clusters = random.randint(3, size=data.shape[0])
     
    # イテレーション
    for _ in range(iteration):
        # 重心を設定
        for c in range(n_cluster):
            new_c = np.array([j for i, j in zip(clusters, data) if i == c]).mean(axis=0)
            centroid[0] = new_c[0]
            centroid[1] = new_c[1]
             
        # 重心に基づいてクラスタを変更
        clusters = np.array([min([(index, (i[0] - j[0])**2 + (i[1] - j[1])**2) for index, j in enumerate(centroid)], key=lambda d : d[1]) for i in data])[:, 0]
         
    return clusters
     
df['cluster'] = k_means(3, df.as_matrix(), 100)
df['cluster_name'] = df.cluster.map(lambda x: 'cluster_{}'.format(int(x)))