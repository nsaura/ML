#!/usr/bin/python
# -*- coding: latin-1 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

#from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam
import keras.backend as K

import keras_utility as ku
ku = reload(ku)


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#-- On reshape la forme des données pour qu'elles soient de la forme 
## "Nombre d'image x 784" 784 = 28**2 qui siginife le nombre de pixe;s
## On donc fait en sorte que chaque pixel soit une caractéristique de l'entrée
# Notons enfin que cette étape n'est nécessaire uniquement à cause de la forme des donnés
#--

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

input_size, hidden_size, code_size = 784, 128, 32
# Avoir une layer code de taille inférieure aux autres empêche l'overfitting
 
input_img = Input(shape=(input_size,)) ## On commence par créer le tenseur des entrées

#-- HL 1 : de type dense connected avec en 1er argument le nombre de nœuds ou sites
## le deuxème est la fonction d'activation.
## Pour l'utilisation de l'autre tensor en argument, on peut regarder la documentation : hidden-1?
## On retient notamment ceci :
##  A `Tensor` can be passed as an input to another `Operation`.
##   This builds a dataflow connection between operations, which
##   enables TensorFlow to execute an entire `Graph` that represents a
##   large, multi-step computation.
hidden_1 = Dense(hidden_size, activation='relu')(input_img) 
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output_img = Dense(input_size, activation='sigmoid')(hidden_2)
# Cette façon de construire le graphe lie les différentes couches.
## On comprend alors que l'autoencoder est une façon élégante de coupler des 
## [Medium encorder ] :
## Dense method is a callable layer, using the functional API we provide it with the input and store the output


autoencoder = Model(input_img, output_img)
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', mean_pred])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=3)

# Output stylée
##Epoch 1/3
##60000/60000 [==============================] - 8s 135us/step - loss: 0.1350
##Epoch 2/3
##60000/60000 [==============================] - 8s 128us/step - loss: 0.0980
##Epoch 3/3
##60000/60000 [==============================] - 8s 132us/step - loss: 0.0926

ku.plot_autoencoder_outputs(autoencoder, 17, (28, 28), x_test)

print ("Taille des poids {}".format(len(autoencoder.get_weights())))
# Output : 8 
## Ce résultat s'explique par le fait que Dense alloue automatiquement des biais. On aura donc une matrice de poids pour de nœuds à nœuds ET une autre pour le biais, et ce pour chaque Dense layer.



