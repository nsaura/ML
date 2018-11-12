#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection    import  train_test_split
from sklearn.linear_model       import  LogisticRegression
from sklearn.datasets           import  load_breast_cancer
cancer = load_breast_cancer()

X,y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
logref1 = LogisticRegression(C=1.).fit(X,y)
logref100 = LogisticRegression(C=100.).fit(X,y)
logref0001 = LogisticRegression(C=0.001).fit(X,y)

print("\x1b[0;30;47m Résultats pour pénalisation sur la norme L2 \x1b[0m")
#plt.figure()
#plt.plot(logref1.coef_.T,'o', label='C=1')
#plt.plot(logref100.coef_.T,'^', label='C=100')
#plt.plot(logref0001.coef_.T,'v', label='C=0.001')

#plt.xticks(range(X.shape[1]), cancer.feature_names, rotation=90)
#plt.hlines(0,0,X.shape[1])
#plt.ylim(-5,5)
#plt.xlabel("Coefficient index")
#plt.ylabel("Coefficient magnitude")
#plt.legend()

print("\n\x1b[0;30;47m Résultats pour pénalisation sur la norme L1 \x1b[0m")

for C, marker in zip([0.001, 1, 100],['o','^', 'v']) :
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train,y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train))) 
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test))) 
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    
plt.xticks(range(X.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0,0,X.shape[1])
plt.ylim(-5,5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend(loc=3)
