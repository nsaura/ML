#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.model_selection    import  train_test_split
from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

import datasets_mglearn as dsets
dsets = reload(dsets)

X,y = make_circles(noise=0.25, factor=0.5, random_state=1)

y_named = np.array(["blue", "red"])[y] #Transforme les 1 de y en "red", les 0 de y en "blue"

X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
    train_test_split(X, y_named, y, random_state=0)
    
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train_named)

print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))

print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))
#Output
#Predicted probabilities:
#[[ 0.01573626  0.98426374]
# [ 0.84335828  0.15664172]
# [ 0.98112869  0.01887131]
# [ 0.97407199  0.02592801]
# [ 0.01352142  0.98647858]
# [ 0.02504637  0.97495363]]
# Nous permet de savoir quelles sont les prédictions qui sont plus ou moins sûres
# Toutefois il faut faire attention au fait que dans un modèle qui OverF, les predictions pourront plus souvent être des faux positifs ou faux négatifs
#Il faut alors calibrer notre modèle. 
# On regardera alors sur un graphe si la prédiction correspond aux proba

fig, axes = plt.subplots(1,2,figsize=(13,5))

dsets.plot_2d_separator(gbrt, X, ax=axes[0], fill=True, alpha=.4, cm=dsets.cm2)
score_images = dsets.plot_2d_scores(gbrt, X, ax=axes[1], cm=dsets.ReBl, function='predict_proba')

for ax in axes:
    dsets.discrete_scatter(X_test[:,0], X_test[:,1], y_test, markers='^', ax=ax)
    dsets.discrete_scatter(X_train[:,0], X_train[:,1], y_train, markers='o', ax=ax)
    
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(score_images, ax=axes.tolist())
axes[0].legend(["Test Class 0", "Test Class 1", "Train Class 0", "Test Class 1"],ncol=4, loc=(.1,1.1))




