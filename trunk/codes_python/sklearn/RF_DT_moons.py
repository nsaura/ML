#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.model_selection    import  train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import datasets_mglearn as dsets
dsets = reload(dsets)

X,y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y, random_state=42)
#Stratify : Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole. For example in a binary classification problem where each class comprises 50% of the data, it is best to arrange the data such that in every fold, each class comprises around half the instances.

forest = RandomForestClassifier(n_estimators=5, random_state=2)
#n_estimators : 5 DT

forest.fit(X_train,y_train)

fig, axes = plt.subplots(2,3,figsize=(20,10)) #Nombre de ligne(s), Nombre de colone(s), taille des fenêtres

for i, (ax,tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    dsets.plot_tree_partition(X_train, y_train, tree, ax=ax)

dsets.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1,-1], alpha=0.4)
axes[-1,-1].set_title("Random forest")
dsets.discrete_scatter(X_train[:,0], X_train[:,1], y_train)

print("RF Training Accuray: {}".format(forest.score(X_train, y_train)))
print("RF Test Accuray: {}".format(forest.score(X_test, y_test)))

