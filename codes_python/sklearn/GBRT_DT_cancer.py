#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.model_selection    import  train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

import datasets_mglearn as dsets
dsets = reload(dsets)

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gbrt_def = GradientBoostingClassifier(random_state=0)
gbrt_def.fit(X_train, y_train)

print("DEFAULT Accuracy on LS: {:.3f}".format(gbrt_def.score(X_train,y_train)))
print("DEFAULT Accuracy on TS: {:.3f}".format(gbrt_def.score(X_test, y_test)))

gbrt_mdlow = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt_mdlow.fit(X_train, y_train)

print("\nMX_D=1 Accuracy on LS: {:.3f}".format(gbrt_mdlow.score(X_train,y_train)))
print("MX_D=1 Accuracy on TS: {:.3f}".format(gbrt_mdlow.score(X_test, y_test)))


gbrt_lr01 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt_lr01.fit(X_train, y_train)

print("\nLR=0.01 Accuracy on LS: {:.3f}".format(gbrt_lr01.score(X_train,y_train)))
print("LR=0.01 Accuracy on TS: {:.3f}".format(gbrt_lr01.score(X_test, y_test)))

n_features = X.shape[1]

fig,axes = plt.subplots(1,3,figsize=(20,10))

for model,ax in zip({gbrt_def, gbrt_mdlow, gbrt_lr01}, axes) :
    ax.barh(xrange(n_features), model.feature_importances_, align='center')
    
    ax.set_xlabel("Feature importances")
    ax.set_ylabel("Feature")
    
    ax.set_title("GBRT avec MD={}, LR={}".format(model.__getattribute__("max_depth"), model.__getattribute__("learning_rate")))

plt.sca(axes[0]) 
plt.yticks(np.arange(n_features), cancer.feature_names)

#plt.sca(axes[-1]) 
#plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
#plt.yticks(np.arange(n_features), cancer.feature_names)

#plt.barh(xrange(n_features), gbrt_mdlow.feature_importances_, align='center',  label="md=1")
#plt.barh(xrange(n_features), gbrt_lr01.feature_importances_, align='center', label='lr0.01')



