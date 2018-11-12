#!/usr/bin/python
# -*- coding: latin-1 -*-
# Ligne au dessous pour éviter des problèmes de syntaxe : accents espaces etc.  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR

import datasets_mglearn as dsets
dsets = reload(dsets)

print("Méthode des moindres carrés : OLS pour Ordinary Least Squares")

# Check chapter 5 to see how to tune parameters
X,y = dsets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

lr = LR().fit(X_train,y_train)

print("\nPrints for \x1b[0;30;47m make wave \x1b[0m") # See https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
print("lr.coef_ {}".format(lr.coef_)) # Numpy Array containing the slop of each W
print("lr.intercept_: {}".format(lr.intercept_)) # Always a float

print("Training Set Score (TSS): {:.2f}".format(lr.score(X_train, y_train)))
print("Prediction Set Score (PSS): {:.2f}".format(lr.score(X_test, y_test)))

#####
X,y = dsets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

lr = LR().fit(X_train, y_train)

print("\nPrints for \x1b[0;30;47m extended boston \x1b[0m")
print("Training Set Score (TSS): {:.2f}".format(lr.score(X_train, y_train)))
print("Prediction Set Score (PSS): {:.2f}".format(lr.score(X_test, y_test)))
print("Overfitting")
