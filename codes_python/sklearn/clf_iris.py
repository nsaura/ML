#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris

iris_dataset = load_iris() #dataset pour commencer. 

# iris_dataset ainsi construite est une sorte de dictionnaire

# les trois possibilités 'setosa', 'versicolor' et 'virginica'se trouvent dans iris_dataset['target_names']

# iris_dataset['data'] contient les différentes classes d'iris avec longueurs et largeurs des pétales et des sépales

# Ces quatre valeurs sont enregistées dans iris_dataset['feature_names']

# iris_dataset['target'] tableau de 0, 1 et 2 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8 )

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) #Utilisation de la méthode des k plus proches voisins. Ici le nombre de voisin considéré est intialisé à 1
knn.fit(X_train, y_train) ## Model base sur la méthode des k plus proches voisins qui contient l'apprentissage

# Making prediction
X_new=np.array([[5,2.9,1,0.2]])

prediction = knn.predict(X_new)
#iris_dataset['feature_names']
#Out[18]: 
#['sepal length (cm)',
# 'sepal width (cm)',
# 'petal length (cm)',
# 'petal width (cm)']


print("Prediction :{}".format(prediction))
#Prediction :[0] Ce qui signifie que la fleur ayant les 4 paramètres de X_new appartient à la classe 0 : 
#iris_dataset['target_names']
#Out[19]: 
#array(['setosa', 'versicolor', 'virginica'], 
#      dtype='|S10')

print("Prediction target name : {}".format(iris_dataset['target_names'][prediction]))
#Prediction target name : ['setosa']

# Evaluating the model

y_pred = knn.predict(X_test)
print("Test set prediction:\n {}".format(y_pred))
#Test set prediction:
# [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]

#y_pred == y_test
#Out[38]: 
#array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#        True,  True,  True,  True,  True,  True,  True,  True,  True,
#        True,  True,  True,  True,  True,  True,  True,  True,  True,
#        True,  True,  True,  True,  True,  True,  True,  True,  True,
#        True, False], dtype=bool)

print("Test set score : {:.2f}".format(np.mean(y_pred==y_test)))
#Test set score : 0.97

# On peut estimer l'incertitude des prédictions avec knn.predict_proba

print("Accuracy on the 6 first terms: \n{}".format(knn.predict_proba(X_test)[:6]))
print("A way to check :\n")
print("Argmax of predicted probabilities: \n{}".format(np.argmax(knn.predict_proba(X_test), axis=1)))
print("Predictions: \n {}".format(knn.predict(X_test)))

print("{}".format(np.argmax(knn.predict_proba(X_test), axis=1) == knn.predict(X_test)))
