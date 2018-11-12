#!/usr/bin/python
# -*- coding: latin-1 -*-
# Ligne au dessus pour éviter des problèmes de syntaxe : accents espaces etc.  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

import datasets_mglearn as dsets
dsets = reload(dsets)

#La méthode Ridge permet de régulariser les profils. L'exigeance supplémentaire est imposée sur la norme euclidienne L2

X,y = dsets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

ridge = Ridge(alpha=1.).fit(X_train,y_train)
print("\n\x1b[0;30;47m Alpha = 0 \x1b[0m")
print("Alpha 1 TSS: {:.2f}".format(ridge.score(X_train, y_train)))
print("Alpha 1 PSS: {:.2f}".format(ridge.score(X_test, y_test)))

print("TSS ridge < TSS OLS \t PSS ridge > PSS OLS")
print("Overfitting limité")

ridge10 = Ridge(alpha=10.).fit(X_train,y_train)
print("\n\x1b[0;30;47m Alpha = 10 \x1b[0m")
print("Alpha 10 TSS: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Alpha 10 PSS: {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("\n\x1b[0;30;47m Alpha = 0.1 \x1b[0m")
print("Alpha 0.1 TSS: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Alpha 0.1 PSS: {:.2f}".format(ridge01.score(X_test, y_test)))

ridge001 = Ridge(alpha=0.01).fit(X_train,y_train)
print("\n\x1b[0;30;47m Alpha = 0.01 \x1b[0m")
print("Alpha 0.01 TSS: {:.2f}".format(ridge001.score(X_train, y_train)))
print("Alpha 0.01 PSS: {:.2f}".format(ridge001.score(X_test, y_test)))
print("\nAlpha mesure la contrainte que l'on impose sur les données \"Force à tendre vers 0\". Plus il est faible, plus on s'approche des moindres carrés")

#dsets.plot_ridge_n_samples()

print("\n\x1b[0;30;47mLasso alpha = 1.0 (default) \x1b[0m")
lasso = Lasso().fit(X_train,y_train)
print("Alpha 1. TSS: {:.2f}".format(lasso.score(X_train, y_train)))
print("Alpha 1. PSS: {:.2f}".format(lasso.score(X_test, y_test)))
print("Coeff non zero: {}".format(np.sum(lasso.coef_ !=0)))

print("\n\x1b[0;30;47mLasso alpha = 0.01 max iter changed into 100000\x1b[0m")
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train,y_train)
print("Alpha 0.01 TSS: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Alpha 0.01 PSS: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Coeff non zero: {}".format(np.sum(lasso001.coef_ !=0)))

print("\n\x1b[0;30;47mLasso alpha = 0.0001 max iter changed into 100000\x1b[0m")
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train,y_train)
print("Alpha 0.0001 TSS: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Alpha 0.0001 PSS: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Coeff non zero: {}".format(np.sum(lasso00001.coef_ !=0)))

## Tracé des coeff en fonction des magnitudes pour différents cas
plt.ion()
plt.plot(lasso.coef_, 's', label="Lasso alpha = 1.0")
plt.plot(lasso001.coef_, '^', label="Lasso alpha = 0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha = 0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha = 0.1")
plt.legend(ncol=2, loc=(0,1.05))
plt.ylim(-25,25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
