#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection    import  train_test_split
from sklearn.linear_model       import  LogisticRegression
from sklearn.datasets           import  make_blobs

import datasets_mglearn as dsets
dsets = reload(dsets)

#Multiclass classification

X,y = make_blobs(random_state=42)
dsets.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Classe 0", "Classe 1", "Classe 2"])

print("\x1b[0;30;47m SVM \x1b[0m")

from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X,y)
print("Coefficient shape: {}".format(linear_svm.coef_.shape))
print("Intercept shape: {}".format(linear_svm.intercept_.shape))

dsets.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
dsets.discrete_scatter(X[:,0], X[:,1], y)
line = np.linspace(-15,15)

for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b','r','g']) :
    plt.plot(line, -(line*coef[0]+intercept)/coef[1], c=color) #Formule provient de One vs Others ?
plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Classe 0', 'Classe 1', 'Classe 2', 'Line classe 0', 'Line classe 1', 'Line classe 2'], loc=(0, 1), ncol=2)




