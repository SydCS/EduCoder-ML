import numpy as np


class smo:
    def __init__(self, max_iter=100, kernel="linear"):
        """
        input:max_iter(int):最大训练轮数
              kernel(str):核函数，等于'linear'表示线性，等于'poly'表示多项式
        """
        self.max_iter = max_iter
        self._kernel = kernel

    # 初始化模型
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        # 将Ei保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 错误惩罚参数
        self.C = 1.0

    # ********* Begin *********#
    # KKT条件
    def _KKT(self, i):
        if self.alpha[i] == 0:
            return self.Y[i] * self._g(i) >= 1
        elif 0 < self.alpha[i] < self.C:
            return np.isclose(self.Y[i] * self._g(i), 1)
        else:  # self.alpha[i] == self.C
            return self.Y[i] * self._g(i) <= 1

    # g(x)预测值，输入xi（X[i]）
    def _g(self, i):
        return np.sum(self.alpha * self.Y * self.kernel(self.X, self.X[i])) + self.b

    # 核函数,多项式添加二次项即可
    def kernel(self, x1, x2):
        if self._kernel == "linear":
            return np.dot(x1, x2.T)
        elif self._kernel == "poly":
            return (np.dot(x1, x2.T) + 1) ** 2
        else:
            raise ValueError("Unsupported kernel type. Must be 'linear' or 'poly'.")

    # E(x)为g(x)对输入x的预测值和y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]

    # 初始alpha
    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if not self._KKT(i):
                E1 = self.E[i]
                # 如果E2是+，选择最小的；如果E2是负的，选择最大的
                if E1 >= 0:
                    j = min(range(self.m), key=lambda x: self.E[x])
                else:
                    j = max(range(self.m), key=lambda x: self.E[x])
                return i, j

    # 选择alpha参数
    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    # 训练
    def fit(self, features, labels):
        """
        input:features(ndarray):特征
              label(ndarray):标签
        """
        self.init_args(features, labels)

        for t in range(self.max_iter):
            i1, i2 = self._init_alpha()
            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = (
                self.kernel(self.X[i1], self.X[i1])
                + self.kernel(self.X[i2], self.X[i2])
                - 2 * self.kernel(self.X[i1], self.X[i2])
            )
            if eta <= 0:
                continue
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (
                self.alpha[i2] - alpha2_new
            )
            b1_new = (
                -E1
                - self.Y[i1]
                * self.kernel(self.X[i1], self.X[i1])
                * (alpha1_new - self.alpha[i1])
                - self.Y[i2]
                * self.kernel(self.X[i2], self.X[i1])
                * (alpha2_new - self.alpha[i2])
                + self.b
            )
            b2_new = (
                -E2
                - self.Y[i1]
                * self.kernel(self.X[i1], self.X[i2])
                * (alpha1_new - self.alpha[i1])
                - self.Y[i2]
                * self.kernel(self.X[i2], self.X[i2])
                * (alpha2_new - self.alpha[i2])
                + self.b
            )
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2
            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

    def predict(self, data):
        """
        input:data(ndarray):单个样本
        output:预测为正样本返回+1，负样本返回-1
        """
        g = np.sum(self.alpha * self.Y * self.kernel(self.X, data)) + self.b
        return np.sign(g).astype(int)

    # ********* End *********#
