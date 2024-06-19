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
