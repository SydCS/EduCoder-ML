## AdaBoost 算法

TODO

```py

```

## sklearn 中的 AdaBoost

```py
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def ada_classifier(train_data, train_label, test_data):
    """
    input:train_data(ndarray):训练数据
          train_label(ndarray):训练标签
          test_data(ndarray):测试标签
    output:predict(ndarray):预测结果
    """
    # ********* Begin *********#
    ada = AdaBoostClassifier(n_estimators=42, learning_rate=1.2)
    ada.fit(train_data, train_label)
    predict = ada.predict(test_data)
    # ********* End *********#
    return predict
```
