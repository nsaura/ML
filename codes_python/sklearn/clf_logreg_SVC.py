#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import datasets_mglearn as dsets
dset = reload(dsets)

X,y = dsets.make_forge()

fig, axes = plt.subplots(1,2,figsize=(10,3)) #Nombre de ligne(s), Nombre de colone(s), taille des fenêtres

for model,ax in zip([LinearSVC(), LogisticRegression()], axes) :
    clf = model.fit(X,y)
    dsets.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    dsets.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

#X,y = dsets.make_blobs(centers=2, random_state=4,n_samples=30)
CC = [0.1,1,1000]
fig, axes = plt.subplots(1,3,figsize=(12,4))

for C,ax in zip(CC,axes) :
    clf=LinearSVC(C=C, tol=0.00001, dual=False).fit(X,y)
    dsets.plot_2d_separator(clf,X,fill=True, eps=0.5, ax=ax, alpha=.7)
    dsets.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    
    ax.set_title("{}; C: {}".format(clf.__class__.__name__, C))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    
axes[0].legend(loc='best')
    
