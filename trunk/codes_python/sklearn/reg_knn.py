import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import datasets_mglearn as dsets
dsets = reload(dsets)

plt.ion()

#dsets.plot_knn_regression(n_neighbors=3)

## The use of KNeighborsRegressor is the same that KNeighborsClassifier

X,y = dsets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("Test set prediction : \n{}".format(reg.predict(X_test)))

print("Test set score : \n{:.2f}".format(reg.score(X_test, y_test)))

#### Analyzing KNeighborsRegressor

fig, axes = plt.subplots(1,3,figsize=(15,4))
#Create a dataset 
line = np.linspace(-3,3,1000).reshape(-1,1) #Convert 1D line into 1 column N ligne matrix
for n, ax in zip([1,3,9], axes) :
    reg = KNeighborsRegressor(n_neighbors = n)
    reg.fit(X_train, y_train)
    
    ax.plot(line, reg.predict(line))
    ax.plot(X_train,y_train, '^', c='b', markersize=8)
    ax.plot(X_test, y_test, 'v', c='r', markersize=8)
    
    ax.set_title(
            "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                n, reg.score(X_train,y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
    
