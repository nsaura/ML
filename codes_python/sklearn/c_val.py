#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
logreg = LogisticRegression()

# X,y = iris.data, iris.target

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)

print("Cross validation scores : {}".format(scores))

print("Average cross-validation score : {}".format(scores.mean()))


from sklearn.model_selection import KFold
kfold = KFold(n_splits=3, shuffle=True, random_state=0)

print("Cross validation - KFold embedded scores : {}".format(\
        cross_val_score(logreg, iris.data, iris.target, cv=kfold)))
        
        
from sklearn.model_selection import ShuffleSplit
sscv    =   ShuffleSplit(test_size=0.5,     \
                         train_size = 0.5,  \
                         n_splits = 10      )
print("SSCV scores : \n{}".format(cross_val_score(\
        logreg, iris.data, iris.target, cv=sscv)))
        
        
X,y = make_blobs(n_samples=12, random_state=0)
from sklearn.model_selection import GroupKFold	
#Assume the first three, second four etc belong to the same group :
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("Goups CV score: \n{}".format(scores))


