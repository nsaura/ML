#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import  tensorflow as tf

try :
    tf.reset_default_graph()
except :
    pass

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

sess = tf.InteractiveSession(config=config)



class Create_Graph() :
    def __init__(self, key, n_hidden = 40, lr=0.001) :
        self.g = tf.Graph()
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.n_hidden = n_hidden
        self.key = key
        self.lr = lr
        
    def create_one_graph(self) :
        with self.g.as_default() as gg :
            with gg.name_scope(self.key) as strscope :

                self.strscope = strscope
                self.x = tf.placeholder(tf.float32, shape=[None, 1], name="%s_state" % self.key)
                hidden = tf.layers.dense(self.x, self.n_hidden, activation=tf.nn.sigmoid,
                                        kernel_initializer = self.initializer, trainable=True)
                hidden = tf.layers.dense(hidden, 1,
                                        kernel_initializer=self.initializer, trainable=True)     
                self.output = tf.nn.sigmoid(hidden)
            
            self.net_params = tf.trainable_variables()
            
            self.action_gradient = tf.placeholder(tf.float32, shape=[None, 1])
            self.actor_gradient  = tf.gradients(self.output, self.net_params,
                                                             -self.action_gradient)
            
            op_grad = tf.train.AdamOptimizer(learning_rate = self.lr)
            self.apply_grad = op_grad.apply_gradients(zip(self.actor_gradient,
                                                          self.net_params))
                                                          
        tf.reset_default_graph()
    
n = 100

dx = np.pi/n

sess_dict = dict()    
g_obj_dict = dict()

for i in range(1, n+1) :
    key = "N%d" % i
    g_obj = Create_Graph(key)
    g_obj.create_one_graph()    
    
    g_obj_dict[key] = g_obj
    with g_obj.g.as_default() as gg : 
        sess = tf.Session(graph=gg)
        init = tf.initialize_all_variables()
        sess.run(init)
            
        sess_dict[key] = sess

x = np.linspace(0, np.pi, n)
X = np.sin(x)

first = np.full((n), 1, dtype=np.float)
current = np.copy(first)

mepoch = 1000

for e in range(mepoch) :
    feeds = []
    for i in range(1, n+1) :
        key = "N%d" % i
        
        g_obj = g_obj_dict[key]
            
        with g_obj.g.as_default() as gg : 
            X_feed  = np.array([current[i-1]]).reshape(1, -1)
            result = sess_dict[key].run(g_obj.output, feed_dict={g_obj.x : X_feed})
            
            feeds.append(X_feed[0][0])
            
        current[i-1] = result[0][0]

    reward = -np.array([(current[j] - X[j]) for j in range(n)])

    derr_reward = [(reward[i+1] - reward[i-1])/(2*dx) for i in range(1, n-1)]

    derr_reward.insert(0, 0)
    derr_reward.insert(len(derr_reward), 0)

    for i in range(1, n+1) :
        key = "N%d" % i
        g_obj = g_obj_dict[key]
        
        with g_obj.g.as_default() as gg :
            action_feed = np.array([derr_reward[i-1]]).reshape(1,-1)
            sess = sess_dict[key]
            
            sess.run(g_obj.apply_grad, feed_dict={g_obj.x:np.array([feeds[i-1]]).reshape(1,-1),
                                                  g_obj.action_gradient:action_feed})
        
    plt.clf()
    plt.plot(x, X)
    plt.plot(x, current)
    plt.pause(0.01)
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
