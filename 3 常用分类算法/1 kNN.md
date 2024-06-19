## 实现 kNN 算法

```py
import numpy as np


class kNNClassifier(object):
    def __init__(self, k):
        """
        初始化函数
        :param k:kNN算法中的k
        """
        self.k = k
        self.train_feature = None  # 用来存放训练数据，类型为ndarray
        self.train_label = None  # 用来存放训练标签，类型为ndarray

    def fit(self, feature, label):
        """
        kNN算法的训练过程
        :param feature: 训练集数据，类型为ndarray
        :param label: 训练集标签，类型为ndarray
        :return: 无返回
        """

        # ********* Begin *********#
        self.train_feature = feature
        self.train_label = label
        # ********* End *********#

    def predict(self, feature):
        """
        kNN算法的预测过程
        :param feature: 测试集数据，类型为ndarray
        :return: 预测结果，类型为ndarray或list
        """

        # ********* Begin *********#
        dists = np.zeros((len(feature), len(self.train_feature)))
        for i in range(len(feature)):
            dists[i] = np.sqrt(np.sum((feature[i] - self.train_feature) ** 2, axis=1))

        preds = np.zeros(len(feature))
        for i in range(len(feature)):
            nearest_indices = np.argsort(dists[i])[: self.k]
            nearest_labels = self.train_label[nearest_indices]
            preds[i] = np.argmax(np.bincount(nearest_labels))
        return preds
        # ********* End *********#
```

## 红酒分类

```py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def classification(train_feature, train_label, test_feature):
    """
    对test_feature进行红酒分类
    :param train_feature: 训练集数据，类型为ndarray
    :param train_label: 训练集标签，类型为ndarray
    :param test_feature: 测试集数据，类型为ndarray
    :return: 测试集数据的分类结果
    """

    # ********* Begin *********#
    scaler = StandardScaler()
    normalized_train_feature = scaler.fit_transform(train_feature)

    # 生成K近邻分类器
    clf = KNeighborsClassifier()
    # 训练分类器
    clf.fit(normalized_train_feature, train_label)
    # 进行预测
    return clf.predict(scaler.transform(test_feature))
    # ********* End **********#
```
