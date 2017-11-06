import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import datasets_mglearn as dsets

dsets = reload(dsets) 

plt.ion()
#generate dataset
X,y = dsets.make_forge()

plt.figure("Blolbs-Classification example")
dsets.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First Feature")
plt.ylabel("Second feature")
print "X.shape: {}".format(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)
print("Test set prediction: {}".format(clf.predict(X_test)))

print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1,3,figsize=(10,3))

for n, ax in zip([1,3,9], axes) :
    clf = KNeighborsClassifier(n_neighbors=n).fit(X,y)
    dsets.plot_2d_classification(clf,X, fill=True, eps = 0.5, ax=ax, alpha=4)
    dsets.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)




