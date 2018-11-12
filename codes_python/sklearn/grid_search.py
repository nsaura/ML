#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import datasets_mglearn as dsets
dsets = reload(dsets)

plt.ion()
iris = load_iris()
X, y = iris.data, iris.target


X_trainval, X_test, y_trainval, y_test    = train_test_split(X, y, random_state=0)
X_train, X_valid, y_train, y_valid  = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training: {}\t size of validation set: {}\t size of test set: {}".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score  =   0.
scores, marker_cv     =   [], []

dict_res    =   dict()
gamma_list  =   [0.001, 0.01, 0.1, 1, 10, 100]
C_list      =   [0.001, 0.01, 0.1, 1, 10, 100]

for i,G in enumerate(gamma_list) :
    gammaa  =   str(G).replace(".", "_")
    dict_res[gammaa] = []    

cpt = 0
for C in C_list :
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100] :
    # for each combinations of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        
        # evaluate the SVC on the test set
        score = svm.score(X_valid, y_valid)
        scores.append(score)
        # if we got a better score, store the score and parameters
        dict_res[str(gamma).replace(".", "_")].append(score)
        
        if score > best_score :
            best_score      =   score
            best_parameters =   {'C': C, 'gamma': gamma}
        cpt += 1 
            
print("Best score: {:.3f}".format(best_score))
print("Best parameters: {}".format(best_parameters))

svm =   SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score on validation set: {:.2f}".format(best_score))
print("Best parameters {}".format(best_parameters))
print("Test score with best parameters: {:.2f}".format(test_score))

#data = pd.DataFrame(X_train, columns=iris.feature_names)
#pd.scatter_matrix(data)

### On peut aller plus vite :
param_grid = {'C': C_list,
               'gamma': gamma_list }
print("Paramet_grid:\n{}".format(param_grid))

##
##
from sklearn.model_selection import GridSearchCV
grid_search =   GridSearchCV(SVC(), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)
print("Test set score: {} ".format(grid_search.score(X_test, y_test)))
print("Params with best cross-validation score: {}".format(grid_search.best_params_))
print("Best estimator :\n{}".format(grid_search.best_estimator_))

results = pd.DataFrame(grid_search.cv_results_)
display(results.head())

scores = np.array(results.mean_test_score).reshape(6,6)

dsets.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'], ylabel='C', yticklabels=param_grid['C'], cmap="viridis")

##
##
param_grid = [{'kernel': ['rbf'], 
                'C': C_list,
                'gamma': gamma_list},
                {'kernel': ['linear'],
                'C' : C_list}
             ]
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cros-validation score: {}".format(grid_search.best_score_))

results =   pd.DataFrame(grid_search.cv_results_)
display(results.T)

##
##
# Nested CV :
from sklearn.model_selection import ParameterGrid, StratifiedKFold
scores = dsets.nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid))
print ("Nested CV scores:\n{}".format(scores))

