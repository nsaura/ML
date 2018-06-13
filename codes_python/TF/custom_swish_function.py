#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import tensorflow as tf

#from tensorflow.python.framework import ops

import matplotlib.pyplot as plt

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

import time
### Define a new activation function ###
## From ##
# https://stackoverflow.com/questions/39921607/how-to-make-a-custom-activation-function-with-only-python-in-tensorflow #

def sigmoid(x) :
    return 1. / (1. + np.exp(-x))
    
def sigmoid_prime(x) :
    return np.exp(-x) / (1. + np.exp(-x))**2
    
def swish(x) :
    return x*sigmoid(x)
    
np_swish = np.vectorize(swish)

def d_swish(x) :
    return x*sigmoid_prime(x) + sigmoid(x)

np_d_swish = np.vectorize(d_swish)


# use float32 before you can convert it to a tensorflow function otherwise tensorflow will complain
np_d_swish_32 = lambda x: np_d_swish(x).astype(np.float32) 
np_swish_32 = lambda x: np_swish(x).astype(np.float32)

# "There is a function in tensorflow tf.py_func(func, inp, Tout, stateful=stateful, name=name) which transforms any numpy function to a tensorflow function"
def tf_d_swish(x,name=None):
    with tf.name_scope(name, "d_swish", [x]) as name:
        y = tf.py_func(np_d_swish_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]
        
# there is a hack to define gradients of a function using tf.RegisterGradient and tf.Graph.gradient_override_map

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# The only thing is that the grad function we need to pass to the above py_func function needs to take a special form. It needs to take in an operation, and the previous gradients before the operation and propagate the gradients backward after the operation.

def swishgrad(op, grad):
    x = op.inputs[0]

    n_gr = tf_d_swish(x)
    return grad * n_gr  

def tf_swish(x, name=None):
    with tf.name_scope(name, "swish", [x]) as name:
        y = py_func(np_swish_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=swishgrad)  # <-- here's the call to the gradient
        return y[0]

if __name__=="__main__":
    #Pour le test
    
    time1 = time.time()
    with tf.Session(config=config) as sess :
        x = tf.constant(np.linspace(-10,10,1000))
        y = tf_swish(x)
        
        sess.run(tf.global_variables_initializer())
        
        time11 = time.time()
        
        grad = tf.gradients(y, [x])[0].eval()
        
        time12 = time.time()
        
        print("Time calculate the gradient = {}".format(time12- time11))
        
#        plt.figure()
#        plt.plot(x.eval(), y.eval())
#        plt.plot(x.eval(), tf.gradients(y, [x])[0].eval())

    time2 = time.time()

    x = np.linspace(-10,10,1000)
    grad = np.gradient(swish(x), x)
    sww = swish(x)
    plt.plot(x, grad)

    time3 = time.time()

    print ("TF part time = {}".format(time2-time1))
    print ("np.grad swish time = {}".format(time3-time2))

