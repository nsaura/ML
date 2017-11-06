#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.model_selection    import  train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

from sklearn.ensemble import RandomForestClassifier

import datasets_mglearn as dsets
dsets = reload(dsets)

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Train Accuracy: {:.3f}".format(forest.score(X_train, y_train)))
print("Train Accuracy: {:.3f}".format(forest.score(X_test, y_test)))

n_features = X.shape[1]
plt.barh(xrange(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel("Feature importances")
plt.ylabel("Feature")

