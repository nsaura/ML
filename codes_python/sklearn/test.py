#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestClassifier

import datasets_mglearn as dsets
dset = reload(dsets)

plt.ion()

X, y = make_moons(n_samples=100, noise=0.25, random_state=0) 

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y)

#dsets.discrete_scatter(XX[:,0], XX[:,1], y)
model = LinearSVC().fit(X_train,y_train)
score_model = model.score(X_test, y_test)
scores = cross_val_score(model, X, y, cv=5)
print("SVM score = {}\n cross_val score = {}".format(score_model, scores)) 

forest = RandomForestClassifier(n_estimators=3, random_state=3, max_depth=3)
forest.fit(X_train,y_train)
print("Forest Accuracy: {:.3f}".format(forest.score(X_test, y_test)))


