#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np

import  tensorflow as tf

try :
    tf.reset_default_graph()
except :
    pass

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

sess = tf.InteractiveSession(config=config)

n_hidden = 40

def create_one_graph(key):
    g = tf.Graph()
    initializer = tf.contrib.layers.variance_scaling_initializer() #Voir plus bas
    
    with g.as_default() as gg :
        with gg.name_scope(key) as strscope :
            x = tf.placeholder(tf.float32, shape=[None, 1])
            hidden = tf.layers.dense(x, n_hidden, activation=tf.nn.sigmoid,
                                     kernel_initializer=initializer, trainable=True)
            hidden = tf.layers.dense(hidden, 1, kernel_initializer=initializer, trainable=True)
            output = tf.nn.sigmoid(hidden)
        
        net_params = tf.trainable_variables()
        print net_params

        action_gradient = tf.placeholder(tf.float32, shape=[None, 1])
        actor_gradient = tf.gradients(output, net_params, -action_gradient)
        op_grad = tf.train.AdamOptimizer(learning_rate=0.01)
        
        op_grad=op_grad.apply_gradients(zip(actor_gradient, net_params))

    tf.reset_default_graph()
    return key, g, op_grad
    
#gr = tf.Graph()
#with gr.as_default() as g :
#    with g.name_scope("gr") as gr_scope :
#        matrix1 = tf.constant([[3., 3.]])
#        matrix2 = tf.constant([[2.],[2.]])
#        product = tf.matmul( matrix1, matrix2, name = "product")

#tf.reset_default_graph()


#gr2 = tf.Graph()
#with gr2.as_default() as g :
#    with g.name_scope("gr2") as gr2_scope :
#        matrix1 = tf.constant([[0., 3.]])
#        matrix2 = tf.constant([[2.],[2.]])
#        product = tf.matmul( matrix1, matrix2, name = "product")

#tf.reset_default_graph()

#def use_a_graph(g, scope):
#    with tf.Session( graph = g ) as sess:
#        tf.initialize_all_variables()
#        result = sess.run( sess.graph.get_tensor_by_name( scope + "product:0" ) )
#        print( result )
#    
