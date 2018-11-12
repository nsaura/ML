#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBC

import datasets_mglearn as dsets
dsets = reload(dsets)

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

gbrt = GBC(max_depth=3, learning_rate=0.01, random_state=0).fit(X_train, y_train)

print("Train Accuracy prediction: {}".format(gbrt.score(X_train, y_train)))
print("Test Accuracy prediction: {}".format(gbrt.score(X_test, y_test)))

print("Prediction Probabilities to detect flase positives or true negatives: \n{}".format(gbrt.predict_proba(X_test)[:6,:]))
print("Sum of these probabilities should be equal to one, each time: \n{}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))

print("We compare y_pred that we know with y_test predicted by the ML: \n{}".format(y_test == gbrt.predict(X_test)))

print("{}".format({k:v for k,v in zip (["False","True"], np.bincount(y_test == gbrt.predict(X_test)))}))

dsets.discrete_scatter(X[:,0], X[:,1], y, markers=['o','^', 'v'])
plt.legend(["{}".format(iris.feature_names[0]), "{}".format(iris.feature_names[1]),"{}".format(iris.feature_names[2])], loc=(0.1,1.1), ncol=4)

plt.ion()
dsets.plot_feature_importances(gbrt, iris)

 
