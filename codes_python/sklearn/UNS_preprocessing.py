#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X,y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

print("Shape of X_train_scaled : {}".format(X_train_scaled.shape))

print("Per-feature minimum before scaling:\n{}".format(X_train.min(axis=0)))
print("Per-feature maximum before scaling:\n{}".format(X_train.max(axis=0)))

print("Per-feature minimum after scaling:\n{}".format(X_train_scaled.min(axis=0)))
print("Per-feature maximum after scaling:\n{}".format(X_train_scaled.max(axis=0)))
print("\n")
#Per-feature minimum before scaling:
#[  6.98100000e+00   1.03800000e+01   4.37900000e+01   1.43500000e+02
#   5.26300000e-02   2.65000000e-02   0.00000000e+00   0.00000000e+00
#   1.16700000e-01   5.02500000e-02   1.14400000e-01   3.60200000e-01
#   7.57000000e-01   6.80200000e+00   2.66700000e-03   3.74600000e-03
#   0.00000000e+00   0.00000000e+00   7.88200000e-03   9.50200000e-04
#   7.93000000e+00   1.24900000e+01   5.04100000e+01   1.85200000e+02
#   8.40900000e-02   4.32700000e-02   0.00000000e+00   0.00000000e+00
#   1.56500000e-01   5.50400000e-02]
#Per-feature maximum before scaling:
#[  2.81100000e+01   3.92800000e+01   1.88500000e+02   2.50100000e+03
#   1.63400000e-01   3.45400000e-01   4.26400000e-01   1.91300000e-01
#   2.90600000e-01   9.57500000e-02   2.87300000e+00   3.64700000e+00
#   2.19800000e+01   5.42200000e+02   3.11300000e-02   1.35400000e-01
#   3.96000000e-01   5.27900000e-02   7.89500000e-02   2.98400000e-02
#   3.60400000e+01   4.95400000e+01   2.51200000e+02   4.25400000e+03
#   2.22600000e-01   1.05800000e+00   1.25200000e+00   2.91000000e-01
#   5.77400000e-01   2.07500000e-01]
#Per-feature minimum after scaling:
#[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
#Per-feature maximum after scaling:
#[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]


X_test_scaled = scaler.transform(X_test)
print("Per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("Per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
#Per-feature minimum after scaling:
#[ 0.07648256 -0.02318339  0.07117684  0.03295864  0.08919383 -0.02232675
#  0.          0.         -0.06152961 -0.00637363 -0.00105126  0.00079104
#  0.00067851  0.00079567 -0.0335172  -0.01134793  0.          0.          0.0233157
# -0.00191763  0.03635717 -0.01268556  0.03107724  0.01349292 -0.09327846
# -0.01574803  0.          0.          0.00023759  0.01252788]
#Per-feature maximum after scaling:
#[ 0.8173127   0.76435986  0.84589869  0.68610817  0.83118173  0.89338351
#  1.00093809  1.05175118  1.07705578  1.03714286  0.50554629  1.37665815
#  0.44117231  0.4224857   0.72596002  0.77972564  0.38762626  0.66054177
#  0.75389768  0.75839224  0.80896478  0.88852901  0.75696001  0.66869839
#  0.9075879   0.81108275  0.61717252  0.88487973  1.20527441  0.77371114]


