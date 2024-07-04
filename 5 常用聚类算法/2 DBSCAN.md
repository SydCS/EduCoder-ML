## DBSCAN 算法流程

和书上伪代码的细节有点不一样

```py
import numpy as np
import random


# 寻找eps邻域内的点
def findNeighbor(j, X, eps):
    N = []
    for p in range(X.shape[0]):  # 找到所有邻域内对象
        dist = np.sqrt(np.sum(np.square(X[j] - X[p])))  # 欧氏距离
        if dist <= eps:
            N.append(p)
    return N


# dbscan算法
def dbscan(X, eps, min_Pts):
    """
    input:X(ndarray):样本数据
          eps(float):eps邻域半径
          min_Pts(int):eps邻域内最少点个数
    output:cluster(list):聚类结果
    """
    # ********* Begin *********#
    core_pts = []  # 核心对象

    for q in range(X.shape[0]):
        if len(findNeighbor(q, X, eps)) >= min_Pts:
            core_pts.append(q)

    k = 0  # 聚类簇数
    cluster = [-1] * X.shape[0]  # 聚类结果

    while len(core_pts) > 0:
        i = random.choice(core_pts)
        cluster[i] = k

        queue = [i]
        while len(queue) > 0:
            q = queue.pop(0)
            neighbor = findNeighbor(q, X, eps)
            if len(neighbor) >= min_Pts:
                core_pts.remove(q)
            for p in neighbor:
                if cluster[p] == -1:
                    cluster[p] = k
                    queue.append(p)

        k += 1

    # ********* End *********#
    return cluster
```

## sklearn 中的 DBSCAN

```py
from sklearn.cluster import DBSCAN


def data_cluster(data):
    """
    input: data(ndarray) :数据
    output: result(ndarray):聚类结果
    """
    # ********* Begin *********#
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    result = dbscan.fit_predict(data)
    return result
    # ********* End *********#
```
