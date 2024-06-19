## 线性支持向量机

```py
from sklearn.svm import LinearSVC


def linearsvc_predict(train_data, train_label, test_data):
    """
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
    output:predict(ndarray):测试集预测标签
    """
    # ********* Begin *********#
    model = LinearSVC(
        penalty="l2",
        loss="squared_hinge",
        dual=False,
        tol=0.001,
        C=0.5,
        multi_class="ovr",
        fit_intercept=False,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    )
    model.fit(train_data, train_label)
    predict = model.predict(test_data)
    # ********* End *********#
    return predict
```

## 非线性支持向量机

```py
from sklearn.svm import SVC


def svc_predict(train_data, train_label, test_data, kernel):
    """
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          kernel(str):使用核函数类型:
              'linear':线性核函数
              'poly':多项式核函数
              'rbf':径像核函数/高斯核
    output:predict(ndarray):测试集预测标签
    """
    # ********* Begin *********#
    if kernel == "linear":
        svc = SVC(kernel="linear")
    elif kernel == "poly":
        svc = SVC(kernel="poly")
    elif kernel == "rbf":
        svc = SVC(kernel="rbf")
    else:
        raise ValueError(
            "Unsupported kernel type. Supported kernels are 'linear', 'poly', 'rbf'."
        )

    svc.fit(train_data, train_label)

    predict = svc.predict(test_data)
    # ********* End *********#
    return predict
```

## 序列最小优化算法

TODO
