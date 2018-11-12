#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.ion()

from sklearn.tree               import  DecisionTreeClassifier 
from sklearn.model_selection    import  train_test_split
from sklearn.datasets import make_moons

import datasets_mglearn as dsets
dsets = reload(dsets)

noise = 0.3
n_samples = 100
X,y = make_moons(n_samples=n_samples, shuffle=True, noise=noise)

class0, class1 = [], []
ax = plt.gca(title="Artisanale")
## Trie dans les classes :
for x in xrange(n_samples) :
    if y[x] == 0 :
        class0.append([X[x,0], X[x,1]])
    else :  
        class1.append([X[x,0], X[x,1]])
#plt.figure("Artisanale")
for i,j in zip(xrange(len(class0)), xrange(len(class1))):
    ax.scatter(class0[i][0], class0[i][1], marker='o', color='red')
    ax.scatter(class1[j][0], class1[j][1], marker='^', color='blue')

child00, child01, child10, child11 = [], [], [], []
sum00, sum01, sum10, sum11 = 0, 0, 0, 0

for i,j in zip(xrange(len(class0)), xrange(len(class1))):
    if np.abs(class0[i][1] - 0.0596) <= 0.0001 :
        child00.append(class0[i])
        sum00 += 1
    else :     
        child01.append(class0[i])
        sum01 +=1

    if np.abs(class1[j][1] - 0.0596) <= 0.0001 :
        child10.append(class1[j])
        sum10 += 1
    else :
        child11.append(class1[j])
        sum11 += 1
xmin = min(X[:,0]) - noise
xmax = max(X[:,0]) + noise

ymin = min(X[:,1]) 
ymax = max(X[:,1]) 

ax.hlines(0.0596, xmin, xmax)

ax.fill_between(np.linspace(xmin,-0.4177, 100), ymin, ymax, facecolor='red', alpha=0.4, interpolate=True, label='First')

#ax.fill_betweenx(np.linspace(ymin,ymax,100), xmin, -0.4177, facecolor='red', alpha=0.2, label='Second')

ax.fill_betweenx(np.linspace(0.0596,ymax,100), -0.4177, 1.1957, facecolor='red', alpha=0.2, label='Third')

ax.fill_betweenx(np.linspace(0.0596,ymax,100), 1.1957, xmax, facecolor='blue', alpha=0.3, label='Second')

ax.fill_between(np.linspace(-0.4177 ,xmax, 100), ymin, 0.0596, facecolor='blue', alpha=0.3, interpolate=True, label='Last')


ax.legend(loc='best')
#plt.figure("Decision Tree")
#dsets.discrete_scatter(X[:,0], X[:,1], y)

#XX = np.zeros((X.shape[0], 1))
#XX[:] = 0.0596

#plt.plot(X[:,0], XX)



###for i in xrange(0, len(prices)):
###    exec("price%d = %s" % (i + 1, repr(prices[i])));###
