#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_trainval, X_test, y_trainval, y_test    = train_test_split(X, y, random_state=0)

X_train, X_valid, y_train, y_valid  = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training: {}\t size of validation set: {}\t size of test set: {}".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0.

for gamma in [0.001, 0.01, 0.1, 1, 10, 100] :
    for C in [0.001, 0.01, 0.1, 1, 10, 100] :
        # for each combinations of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        
        # evaluate the SVC on the test set
        score = svm.score(X_valid, y_valid)
        # if we got a better score, store the score and parameters
        if score > best_score :
            best_score      =   score
            best_parameters =   {'C': C, 'gamma': gamma}
         
print("Best score: {:.3f}".format(best_score))
print("Best parameters: {}".format(best_parameters))

svm =   SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters {}".format(best_parameters))
print("Test score with best parameters: {:.2f}".format(test_score))

data = pd.DataFrame(X_train, columns=iris.feature_names)
pd.scatter_matrix(data)

plt.show()




