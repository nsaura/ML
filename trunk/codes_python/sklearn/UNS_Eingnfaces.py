#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

plt.ion()

verbose = False 
if verbose == True :
    fig, axes = plt.subplots(2,5, figsize=(15,8), 
                            subplot_kw={'xticks': (), 'yticks': ()})

    for target, image, ax in zip(people.target, people.images, axes.ravel()) :
        ax.imshow(image)
        ax.set_title(people.target_names[target])
        
print("people.image.shape {}".format(people.images.shape))
print("Number of classes {}".format(len(people.target_names)))

#In [61]: X_train.shape
#Out[61]: (1547, 5655)

# Count how many photos of people we have
counts = np.bincount(people.target)

for i, (count, name) in enumerate(zip(counts, people.target_names)) :
    print "{0:25} {1:3} \n".format(name, count), 

#dataset skewed (biaisée)
#George W Bush             530
#Colin Powell              236

# To make it less skewed use np.unique

mask = np.zeros(people.target.shape, dtype = np.bool)

for target in np.unique(people.target ) :
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

# Très shadé ! Prends les valeurs de 0 ou 1 (np.unique sort un tableau d'élements disctints qui apparaissent) puis construit un tableau, mask, qui contient les indices pour lesquelles on a des target 0 ou 1 ou autres
# On peut ensuite évaleur les valeurs de x1 et x2 en ces indices et les colorier avec la couleur correspondant à la classe ainsi testée

# Scale the Greyscale values to be between 0 and 1 istead of 0-255 for better numeric stability
X_people /= 255.0

# We want to construct of a 1 nearest-neighbor classifier

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify = y_people, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
#print("Test set score of 1-nn {:.2f}".format(knn.score(X_test, y_test)))
# Train score is 1.0 ---> overfitting


### On va plutôt utiliser des méthodes non supervisées --> PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=100,whiten=True,random_state=0).fit(X_train) #Pas besoin de labels
X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)

print("X_train_pca.shape{}".format(X_train_pca.shape))
#(1547, 100)

#We use the new classification in our classifier
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_pca, y_train)
print("Test set accuracy nn-1 {}".format(knn.score(X_test_pca, y_test)))
#31% on a gagné 10 points

print("pca.components_.shape : {}".format(pca.components_.shape))

verbose2 = True 
if verbose2 == True :
    fig, axes = plt.subplots(3,5, figsize=(15,12), 
                            subplot_kw={'xticks': (), 'yticks': ()})

    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())) :
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title("{} component".format(i+1))



