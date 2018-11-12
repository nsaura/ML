#!/usr/bin/python
# -*- coding: latin-1 -*-

from sklearn.model_selection    import  train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier

plt.ion()

cancer = load_breast_cancer()
print("Cancer data per-feature maxima: \n{}".format(cancer.data.max(axis=0)))

X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

mlp = MLPClassifier(random_state=42).fit(X_train, y_train)

print("Accuracy on the training set: {:.3f}".format(mlp.score(X_train,y_train))) 
print("Accuracy on the test set: {:.3f}".format(mlp.score(X_test, y_test)))

#Output
#Accuracy on the training set: 0.906
#Accuracy on the test set: 0.881

#### On doit rescaler les données pour satisfaire les exigeances du MLP (moyenne à 0  et std à 1) et améliorer les capacités du MLP


X_train_scaled = (X_train-X_train.mean(axis=0))/X_train.std(axis=0)
#### On utilise de même X_train.mean pour rescaler les données

X_test_scaled = (X_test-X_train.mean(axis=0))/X_train.std(axis=0)

mlp = MLPClassifier(random_state=42, max_iter=1000).fit(X_train_scaled, y_train)

print("\nAccuracy on the Scaled training set: {:.3f}".format(mlp.score(X_train_scaled,y_train))) 
print("Accuracy on the Scaled test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

print("\nOn passe alpha de {:.4f}, à {:.4f}".format(mlp.__getattribute__("alpha"), 1))
mlp = MLPClassifier(random_state=42, max_iter=1000, alpha=1).fit(X_train_scaled, y_train)

print("Accuracy on the Scaled training set: {:.3f}".format(mlp.score(X_train_scaled,y_train))) 
print("Accuracy on the Scaled test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

fig, axes = plt.subplots(2,1, figsize=(15,8))

for ax, alpha in zip(axes, [1, 10]) :
    mlp = MLPClassifier(random_state=42, max_iter=1000,alpha=alpha).fit(X_train_scaled, y_train)
    ax.set_title("alpha = {:.4f}".format(alpha), loc=u'right')
    ax.imshow(mlp.coefs_[0], cmap='viridis', interpolation='none') 
    ax.set_xlabel("Columns in weights Matrix")
    ax.set_ylabel("Input features")
plt.sca(axes[0]) 
plt.yticks(xrange(X.shape[1]), cancer.feature_names)
plt.sca(axes[-1]) 
plt.yticks(xrange(X.shape[1]), cancer.feature_names)
plt.colorbar()


