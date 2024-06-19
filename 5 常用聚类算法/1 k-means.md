## 距离度量

$$
\| x \|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{\frac{1}{p}}
$$

```py
import numpy as np


def distance(x, y, p=2):
    """
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离，等于2时为欧氏距离
    output:distance(float):x到y的距离
    """
    # ********* Begin *********#
    return np.power(np.sum(np.power(np.abs(x - y), p)), 1 / p)
    # ********* End *********#
```

## 什么是质心

```py
import numpy as np


# 计算样本间距离
def distance(x, y, p=2):
    """
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离，等于2时为欧氏距离
    output:distance(float):x到y的距离
    """
    # ********* Begin *********#
    return np.linalg.norm(x - y, p)
    # ********* End *********#


# 计算质心
def cal_Cmass(data):
    """
    input:data(ndarray):数据样本
    output:mass(ndarray):数据样本质心
    """
    # ********* Begin *********#
    Cmass = np.mean(data, axis=0)
    # ********* End *********#
    return Cmass


# 计算每个样本到质心的距离，并按照从小到大的顺序排列
def sorted_list(data, Cmass):
    """
    input:data(ndarray):数据样本
          Cmass(ndarray):数据样本质心
    output:dis_list(list):排好序的样本到质心距离
    """
    # ********* Begin *********#
    dis_list = sorted([distance(x, Cmass) for x in data])
    # ********* End *********#
    return dis_list
```

## k-means 算法流程

```py
import numpy as np


# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


class Kmeans:
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛
    """

    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        np.random.seed(1)

    # ********* Begin *********#
    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        return X[np.random.choice(X.shape[0], self.k, replace=False)]

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distance = euclidean_distance(sample, centroids)
        return np.argmin(distance)

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        return np.array([self._closest_centroid(sample, centroids) for sample in X])

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        return np.array([X[clusters == c].mean(axis=0) for c in range(self.k)])

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        return clusters

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)
        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for i in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            # 计算新的聚类中心
            new_centroids = self.update_centroids(clusters, X)
            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            if np.linalg.norm(new_centroids - centroids) < self.varepsilon:
                break
            centroids = new_centroids
        return self.get_cluster_labels(clusters, X)

    # ********* End *********#
```

## sklearn 中的 k-means

```py
from sklearn.cluster import KMeans


def kmeans_cluster(data):
    """
    input:data(ndarray):样本数据
    output:result(ndarray):聚类结果
    """
    # ********* Begin *********#
    km = KMeans(n_clusters=3, random_state=888)
    result = km.fit_predict(data)
    # ********* End *********#
    return result
```
