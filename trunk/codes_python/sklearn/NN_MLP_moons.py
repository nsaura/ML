#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.model_selection    import  train_test_split
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

import datasets_mglearn as dsets
dsets = reload(dsets)

X,y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10,10]).fit(X_train, y_train)
plt.figure("[10,10]")
dsets.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
dsets.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

print("hidden_layer_sizes : [10] : 1 hidden avec 10 nœuds \n \t \t [10,10] : 2 hidden 10 nœuds chacune")

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10,10,10]).fit(X_train, y_train)
plt.figure("{}".format(mlp.__getattribute__("hidden_layer_sizes"    )))
dsets.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
dsets.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


mlp = MLPClassifier(solver='lbfgs', activation='tanh',random_state=0, hidden_layer_sizes=[10,10,10]).fit(X_train, y_train)
plt.figure("{}, {}".format(mlp.__getattribute__("hidden_layer_sizes"), mlp.__getattribute__("activation")))
dsets.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
dsets.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


fig, axes = plt.subplots(2,4,figsize=(20,8)) 
for ax, n_hidden_nodes in zip(axes, [10,100]) :
    for axx, alpha in zip(ax, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], random_state=0).fit(X_train, y_train)
        dsets.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=axx)
        dsets.discrete_scatter(X_train[:,0], X_train[:,1], y_train, ax=axx)
        axx.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
        
        
