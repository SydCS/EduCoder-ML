## PCA 算法流程

```py
import numpy as np


def pca(data, k):
    """
    对data进行PCA，并将结果返回
    :param data:数据集，类型为ndarray
    :param k:想要降成几维，类型为int
    :return: 降维后的数据，类型为ndarray
    """

    # ********* Begin *********#
    # 计算样本各个维度的均值
    mean = np.mean(data, axis=0)
    # demean
    after_demean = data - mean

    # 计算after_demean的协方差矩阵
    # after_demean的行数为样本个数，列数为特征个数
    # 由于cov函数的输入希望是行代表特征，列代表数据的矩阵，所以要转置
    cov = np.cov(after_demean.T)

    # eig为计算特征值与特征向量的函数
    # cov为矩阵，value为特征值，vector为特征向量
    value, vector = np.linalg.eig(cov)

    # 根据特征值value将特征向量vector降序排序
    # 筛选出前k个特征向量组成映射矩阵P
    p = vector[:, np.argsort(value)[::-1][:k]]
    # after_demean和P做矩阵乘法得到result
    return after_demean.dot(p)
    # ********* End *********#
```

## sklearn 中的 PCA

```py
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


def cancer_predict(train_sample, train_label, test_sample):
    """
    使用PCA降维，并进行分类，最后将分类结果返回
    :param train_sample:训练样本, 类型为ndarray
    :param train_label:训练标签, 类型为ndarray
    :param test_sample:测试样本, 类型为ndarray
    :return: 分类结果
    """

    # ********* Begin *********#
    pca = PCA(n_components=11)
    train_sample_transformed = pca.fit_transform(train_sample)
    test_sample_transformed = pca.transform(test_sample)

    clf = LinearSVC()
    clf.fit(train_sample_transformed, train_label)
    return clf.predict(test_sample_transformed)
    # ********* End *********#
```
