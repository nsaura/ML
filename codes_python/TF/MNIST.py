#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 

# one hot encoding means 
# 7 : 0000000100
# 2 : 0010000000

sess = tf.InteractiveSession()

#Build Computational graph

x = tf.placeholder(tf.float32, shape=[None,784]) # 2D Tensors
# First Dimension correspond of : any size, no restriction
# 784 = 28x28 pixels mnist images

y = tf.placeholder(tf.float32, shape = [None, 10]) 
# 10 class corresponding to 10 digits between 0 and 9
# Desired values

w = tf.Variable(tf.zeros([784, 10]))
# 784 because 784 input and 10 possibles outputs
b = tf.Variable(tf.zeros([10]))
# 10 classes

sess.run(tf.global_variables_initializer())

# Linear regression
y_ = tf.matmul(x,w) + b
# y_ prediction labels, to be compared with y 

# Loss function for LinReg is Soft Max Cross Entropy 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
# 
## Train the module 

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

## cross entropy l'entropie croisée entre deux lois de probabilité mesure le nombre de bits moyen nécessaires pour identifier un événement issu d'un espace probabiliste, si l'encodage est basé sur une loi de probabilité q {\displaystyle q} q, plutôt que sur une "vraie" loi p {\displaystyle p} p.

for _ in range(1000) :
   batch = mnist.train.next_batch(100) #Each iteration we take 100 images with 100 corresponding labels. Input will be in batch[0] and corresponding labels will be in batch[1]
   train_step.run(feed_dict={x:batch[0], y:batch[1]})
   
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
# We chek if the y_, predicted labels, match the desire prediction

#print(correct_prediction.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels})) 
# Output a tab that testing y_ == y 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# We evaluate the accuracy on the test data
# tf.cast to change the type "Casts a tensor to a new type"

print(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))

#Final Output  : 0.9198

