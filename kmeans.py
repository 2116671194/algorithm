import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 计算两个坐标的曼哈顿距离
def manhattanDistance(a: tuple, b: tuple) -> float:
    distances = 0.
    for _a, _b in zip(a, b):
        distances += math.fabs(_a - _b)
    return distances

# 计算两个坐标的欧式距离
def euclidean_Distance(a: tuple, b: tuple) -> float:
    distances = 0.
    for _a, _b in zip(a, b):
        distances += (_a - _b) ** 2
    return math.sqrt(distances)

# 二维聚类可视化
def display_2D_cluster(colors: list, cluster_rs: dict):
    keys = list(cluster_rs.keys())
    for i in range(len(colors)):
        cs = np.array(cluster_rs[keys[i]])
        plt.scatter(cs[:,0], cs[:,1], color=colors[i], marker='*')
    plt.scatter(np.array(keys)[:, 0], np.array(keys)[:, 1], marker='o', s=50, color='black')
    plt.show()
    
# 三维聚类可视化
def display_3D_cluster(colors: list, cluster_rs: dict):
    ax = plt.axes(projection='3d')
    keys = list(cluster_rs.keys())
    for i in range(len(colors)):
        cs = np.array(cluster_rs[keys[i]])
        ax.scatter3D(cs[:,0], cs[:,1], cs[:,2], color=colors[i], marker='*')
    ax.scatter3D(np.array(keys)[:, 0], np.array(keys)[:, 1], np.array(keys)[:, 2], marker='o', s=100, color='black')
    plt.show()


# kmeans聚类算法
def kmeans(K: int=5,
          coors: list=[(2.0, 1.0, 2.0), (1.0, 2.0, 1.0), (2.0, 2.0, 2.0), 
                       (2.0, 3.0, 3.0), (3.0, 2.0, 2.0), (3.0, 3.0, 3.0), 
                       (2.0, 4.0, 4.0), (3.0, 5.0, 5.0), (4.0, 4.0, 4.0), 
                       (5.0, 3.0, 3.0)],
          distance_function: object=euclidean_Distance,
          init_cluster_centers = None) -> dict:
    """
    @author: drq
    @params: K: 聚类数
    @params: coors: 样本数，[(,,,),(,,,),(,,,),...,(,,,)]
    @params: distance_function: 距离函数
    @params: init_cluster_centers: 初始化聚类质心
    @return: 聚类结果：dict
    """
    # 初始化聚类中心
    if init_cluster_centers is None:
        init_cluster_centers = []
        
        # 保存聚类结果
        cluster_rs = dict()
        for i in random.sample(range(0, len(coors)), K):
            cluster_rs[coors[i]] = []
            init_cluster_centers.append(coors[i])
        print('初始聚类中心：', init_cluster_centers)
    else:
        # 保存聚类结果
        cluster_rs = dict()
        for cluster_center in init_cluster_centers:
            cluster_rs[cluster_center] = []
        print('初始聚类质心：', init_cluster_centers)
    flags = 1
    # 开始训练
    while True:
        # 计算每个样本与聚类中心的欧式距离
        for coor in coors:
            m = float('inf')
            for cluster_center in init_cluster_centers:
                distance = euclidean_Distance(coor, cluster_center)
                if distance < m:
                    m = distance
                    c_c = cluster_center
            cluster_rs[c_c].append(coor)
            
        # 可视化展示
        display_3D_cluster(['red', 'green', 'blue', 'yellow', 'pink'], cluster_rs)
        
        # 调整聚类中心
        init_cluster_centers.clear()
        for cluster_center, cluster_sample in cluster_rs.items():
            cluster_sample = np.array(cluster_sample)
            cluster_sample = tuple(cluster_sample.mean(0))
            init_cluster_centers.append(cluster_sample)
        print(f'第{flags}次训练聚类中心：', init_cluster_centers)
        # 判别当前聚类中心与上一次聚类中心是否相同，相同停止训练，不同继续
        if list(cluster_rs.keys()) == init_cluster_centers:
            break
        else:
            flags += 1
            cluster_rs.clear()
            for cluster_center in init_cluster_centers:
                cluster_rs[cluster_center] = []
            
    return cluster_rs