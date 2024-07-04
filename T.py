from sklearn.svm import SVR


def svr_predict(train_data, train_label, test_data):
    """
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
    output:predict(ndarray):测试集预测标签
    """
    # ********* Begin *********#
    svr = SVR(kernel="rbf", C=100, gamma=0.001, epsilon=0.1)

    svr.fit(train_data, train_label)

    predict = svr.predict(test_data)
    # ********* End *********#
    return predict
