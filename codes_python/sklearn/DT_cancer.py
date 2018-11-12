#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.tree               import  DecisionTreeClassifier, export_graphviz 
from sklearn.model_selection    import  train_test_split
from sklearn.datasets           import  load_breast_cancer
cancer = load_breast_cancer()

import datasets_mglearn as dsets
dsets = reload(dsets)

import graphviz

def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1]
    plt.barh(xrange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importances")
    plt.ylabel("Feature")
##---###    

def get_tree(tree, **kwargs):
    from io import BytesIO 
    f = BytesIO()
    print ("type(f): {}".format(type(f)))
    export_graphviz(tree, f, **kwargs)
    fi = open("mytree.dot","r")
    dot_file = f.read()
    fi.close()
    graphviz.Source(fi.getvalue())
    
    
##--------------------------------------------------------
X,y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)

print("Train accuracy: {:.3f}".format(tree.score(X_train,y_train)))

print("Test accuracy: {:.3f}".format(tree.score(X_test,y_test)))

print("\n\x1b[0;30;47m max_depth = 4 \x1b[0m")
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train,y_train)

print("MD=4 Train accuracy: {:.3f}".format(tree.score(X_train,y_train)))

print("MD=4 Test accuracy: {:.3f}".format(tree.score(X_test,y_test)))

s=export_graphviz(tree, out_file='tree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

f = open('tree.dot', "r")
dot_graph = f.read()
f.close()
graphviz.Source(dot_graph, format='pdf', filename='tree.pdf')
graphviz.view(dot_graph)

print("\n\x1b[0;30;47mPour voir le graphe, aller sur http://webgraphviz.com/, copier l'affichage en balise\x1b[0m")


print("Feature Importance:\n{}".format(tree.feature_importances_))

plot_feature_importances_cancer(tree)

from IPython.display import display

tree = dsets.plot_tree_not_monotone()
display(tree)




