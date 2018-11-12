#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


x = [6,2]
h = [1,2,5,4]

y = np.convolve(x,h,"full")

from scipy import signal as sg

I= [[255,   7,  3],
    [212,   240,4],
    [218, 216,  230]] 
g = [[-1,1]] # 2D

print("Without zero padding: \n {0}".format(sg.convolve(I, g, "valid")))

print("With zero padding: \n {0}".format(sg.convolve(I, g, "full")))


print("\x1b[0;30;47m With Tensorflow \x1b[0m")
import tensorflow as tf

# Building Graph
input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))

op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding="VALID") # Output size = Input size - kernel/filter dim + 1

op2 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding="SAME") #Output same size of the input 

# Initialization
init = tf.global_variables_initializer()
with tf.Session() as sess :
    sess.run(init)
    
    print("Input")
    print("{0}\n".format(input.eval()))
    print("Filter/Kernel")
    print("{0}\n".format(filter.eval()))
    
    print("Results with valid condition")
    result = sess.run(op); print (result, "\n")
    
    print("Result with SAME condition")
    result = sess.run(op2); print(result, "\n")
    
    
    
    
    
