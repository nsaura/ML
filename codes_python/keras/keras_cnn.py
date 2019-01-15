#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import sys
import numpy as np
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

#Init signature: Conv2D(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)

input_shape = (7,7,1)
strides = (1,1)
feature_maps = 64
kernel_size = 5

padding = 'same'

new_steps = [(input_shape[i] - kernel_size)/strides[i] + 1 for i in range(2)]

print new_steps

for j, n in enumerate(new_steps) :
    if (input_shape[j] - kernel_size) % strides[j] != 0 :
        print ("For %d, there sould be a problem" % j)

#add model layers
model.add(Conv2D(feature_maps, kernel_size=kernel_size, activation='relu', strides=strides, padding='same', input_shape=input_shape))
# (None, 26, 26, 64)

print "Output Convo2D shape : {}".format(model.output_shape)

#model.add(Conv2D(32, kernel_size=3, activation='relu'))
# (None, 24, 24, 32)

model.add(MaxPooling2D(pool_size=2, padding='same'))

print "Output Maxpooling2D shape : {}".format(model.output_shape)
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
#   La convolution sort une matrice de taille (None, new_steps, new_steps, 64) sans padding 
#   
#   Strides = (1,1)
#   Lorsque kernel_size = 2 : new_steps = 28-2 + 1 = 27
#   Lorsque kernel_size = 3 : new_steps = 28-3 + 1 = 26
#   Lorsque kernel_size = 4 : new_steps = 28-4 + 1 = 25
#   
#   Strides = (2,2)
#   Lorsque kernel_size = 2 : new_steps = (28-2)/2 + 1 = 14
#   ETC
#   
#
#   Lorsque padding = 'same', output_shape = input_shape (Inconsistent lorsque Strides > (1,1))
#   c'est le zéro padding (Vérifier)
#
#   La formule de new_steps devient : (n_steps-kernel_size + 2*padding)/strides + 1
#
#   Exemple
#   kernel_size = 3, padding='same', strides=(1,1), feature_maps = 64
#   new_steps = 28 = (28 - 3 + 2*p) + 1 => p = 1, i.e. une ligne de 0 tout en haut et bas
#   La sortie est effectivement (None, 28, 28, 64)
#   
#   Supposons maintenant kernel_size = 6, strides et feature_maps identiques   
#   new_steps = 28 = (28 - 5 + 2*p) + 1 => p = 2, i.e. deux lignes de 0 tout en haut et bas
#
#   On fera bien ce calcul avant pour savoir si p entier. (e.g. kernel_size = 6 => p = 2.5 pb)
#   L'erreur n'est pas affichée et la sortie à la forme valide, mais traitement nécessaire.
#
#
#   Voir DL page 344 Fig. 9.10. Il est possible que la convolution soit fait sur des zones plus petites (a cause du strides) 
#   Le Pooling réduit la taille de feature 
#
#   Le flatten "empile" toutes les données dans une matrice (None, filters*new_steps*autres_dim)
#
#   La couche dense aura une sortie de la taille de n_neurons_dense_layer

#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# A propos du padding dans max pool et en règle général :
#
# Voir https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
# Reponse de Shital Shah
#
# VALID: Don't apply any padding, i.e., assume that all dimensions are valid so that input image fully gets covered by filter and stride you specified.
# SAME: Apply padding to input (if needed) so that input image gets fully covered by filter and stride you specified. For stride 1, this will ensure that output image size is same as input.

# Exemple :
# 1 - Maxpooling Padding = valid (suppose que les dimensions sont bonnes)
#   input_shape = (7,7,1)
#   model.add(Conv2D(feature_maps, kernel_size=kernel_size, activation='relu', strides=strides, padding='same', input_shape=input_shape))
#   model.add(MaxPooling2D(pool_size=2, padding='valid'))
#------
#   Output Convo2D shape : (None, 7, 7, 64)
#   Output Maxpooling2D shape : (None, 3, 3, 64)
#
#
# 2 - Maxpooling Padding = same (auto pads pour que toute l'image soit parcourue par le filtre en fonction du stride
#   input_shape = (7,7,1)
#   model.add(Conv2D(feature_maps, kernel_size=kernel_size, activation='relu', strides=strides, padding='same', input_shape=input_shape))
#   model.add(MaxPooling2D(pool_size=2, padding='same'))
#------
#   Output Convo2D shape : (None, 7, 7, 64)
#   Output Maxpooling2D shape : (None, 4, 4, 64)

# On voit alors que les dimensions n'étaient pas bonnes et en "valid" des informations ont sûrement été perdues
# Olivier Moindrot (accepted answer) précise que "max pool with 2x2 kernel, stride 2 and SAME padding (this is the classic way to go)"



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



