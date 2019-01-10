#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.ion()

#plot the first image in the dataset
#plt.imshow(X_train[0])

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

from keras.utils import to_categorical

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
# (None, 26, 26, 64)

model.add(Conv2D(32, kernel_size=3, activation='relu'))
# (None, 24, 24, 32)

model.add(MaxPooling2D(pool_size=4))
print model.output_shape 
sys.exit()


#In between the Conv2D layers and the dense layer, there is a 'Flatten' layer. Flatten serves as a connection between the convolution and dense layers.
model.add(Flatten())
# (None, 18432) 24*24*32 = 18432

model.add(Dense(10, activation='softmax'))
# (None, 10)

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


###


# 26 = 28-3 + 1
# dim (batch, new_rows, new_cols, filters)
#-----------------------------------------#
####   Notes sur Dimension des poids   ####
#-----------------------------------------#
#
# --> voir model.get_weights() 
# 
#
# A = np.array(model.get_weights()), A.shape = (6,), 6 matrices de matrices (parfois) ou matrices simples

# A[0]
#   
#   La couche Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features) générera des poids de la taille (kernel_size, n_features, filters)
#
#
# A[1]
#   
#   La couche de pooling (avec le flatten qui effectue  va évaluer sur une zone pool_size x pool_size la max (ici) et générer une couche de poids = filters de la taille (filters,) 
#
#
# A[2] et A[3]
# 
#   La couche Dense lie la couche précédente avec la taille demandée en premier argument de la fonction. Ainsi ici les poids de la première couche dense sera un matrice de taille (filters, n_neurons_dense_layer)
#   A cette couche de poids, s'ajoute les biais, un vecteur de taille (n_neurons_dense_layer)
#
#
# A[4] et A[5]
#   Idem pour la couche suivant, on aura une matrice de poids (n_neurons_dense_layer, n_output)
#   Et donc un vecteur de biais (n_output)
#
#
#  Au total on aura sum(A[i].size for i in range(A.size)) = 3621 paramètres à optimiser


#------------------------------------------------#
####   Notes sur les dimensions des couches   ####
#------------------------------------------------#
#
# --> voir model.output_shape sur chaque couche  #
#
#
# L'entrée est toujours de taille (n_steps, n_features)
#
# On note new_steps = (n_steps-kernel_size)/strides + 1
#
#
#   La convolution sort une matrice de taille (None, new_steps, 64) sans padding 
#
#   Le Pooling réduit la taille de feature 
#
#   Le flatten "empile" toutes les données dans une matrice (None, filters*new_steps*autres_dim)
#
#   La couche dense aura une sortie de la taille de n_neurons_dense_layer

