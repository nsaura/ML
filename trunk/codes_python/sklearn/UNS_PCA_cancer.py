#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import datasets_mglearn as dsets
dsets = reload(dsets)

cancer = load_breast_cancer()

X,y = cancer.data, cancer.target

plt.ion()

malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

verbose = False
if verbose == True :
    fig, axes = plt.subplots(15, 2, figsize = (15,30))
    ax = axes.ravel()
    ## Visualisation des features et leur appartenance, sur des histogrammes
    for i in range(30) :
        _ , bins = np.histogram(X[:,i], bins=50)
    #    bins : int or sequence of scalars or str, optional
    #    If `bins` is an int, it defines the number of equal-width
    #    bins in the given range (10, by default). If `bins` is a
    #    sequence, it defines the bin edges, including the rightmost
    #    edge, allowing for non-uniform bin widths.

        ax[i].hist(malignant[:, i], bins=bins, color=dsets.cm3(1), alpha=0.5)
        ax[i].hist(benign[:, i], bins=bins, color=dsets.cm2(0), alpha=0.5)
        ax[i].set_title(cancer.feature_names[i], fontdict={'verticalalignment': 'baseline',
                                                             'horizontalalignment': "left"})
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant","benign"], loc="best")
    
## Utilisation de La PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2) # Ici on n'en garder que 2
pca.fit(X_scaled)
#Ou bien  pca = PCA(n_components=2).fit(X_scaled)

X_pca = pca.transform(X_scaled)
print("Original shape {}".format(str(X_scaled.shape)))
print("PCA shape {}".format(str(X_pca.shape)))
#In [70]: run UNS_PCA_cancer.py
#Original shape (569, 30)
#PCA shape (569, 2) meaning that we only kept 2 features

#plt first vs second components
if verbose == True :
    plt.figure(figsize=(8,8))
    dsets.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
    plt.legend(cancer.target_names, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("First Principal component")
    plt.ylabel("Second Principal component")

print("PCA components:\n {}".format(pca.components_))

## Heat map
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0,1], ['First component','Second component'])
plt.colorbar()
plt.xticks(xrange(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("features")
plt.ylabel("Principal components")













                                    

