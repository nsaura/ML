#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

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

y_ = tf.nn.softmax(tf.matmul(x,w) + b)
# y_ prediction labels, to be compared with y 

#Soft Max Cross Entropy 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) ## It looks like cross entropy with the softmax as y_ = softmax...

## Train the module 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


