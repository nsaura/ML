#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
#from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge

from keras.layers.normalization import BatchNormalization as BN
#Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

from keras.optimizers import Adam

import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 800
HIDDEN2_UNITS = 400

def act(x):
        return 1.67653251702 * (x * K.sigmoid(x) - 0.20662096414)

def tanh(x):
        return (K.tanh(x) + 1) / 2

#####################
#####################
### Actor Network ###
#####################
#####################

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = tf.placeholder(tf.float32,shape=[]) #LR decay

        K.set_session(sess)

        #Now create the model
        #create actor network outs the model, weiths and inputs
        
        # First
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        
        # Target
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        # Define the place where the gradient will be apply 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        
        # New gradient values definition : \frac{\partial Out} {\partial w}
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        
        # pairs params and weights
        grads_and_vars = zip(self.params_grad, self.weights)
        
        self.optimize=tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE). apply_gradients(grads_and_vars)

        self.sess.run(tf.global_variables_initializer())
        # Nodes closed

    def train(self, states, action_grads, learning_rate):
            
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,
            self.LEARNING_RATE : learning_rate
                                                })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        # Formule algo "Continuous control with deep reinforcement learning"
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        
        self.target_model.set_weights(actor_target_weights) #Eqvuivalent to assign from tensorflow
    
    def create_actor_network(self, state_size,action_dim):
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='selu', kernel_initializer = 'normal')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal')(h0)
        V = Dense(action_dim ,activation=tanh, kernel_initializer = 'normal')(h1)
        model = Model(inputs=S,outputs=V)
        model.summary()
    return model, model.trainable_weights, S

######################
######################
## Critique Network ##    
######################
######################

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  

        # Here action is not negative
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
    
    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='selu', kernel_initializer = 'normal')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal')(w1)
        h2 = concatenate([h1,a1])
        h3 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal')(h2)
        V = Dense(1,activation='linear')(h3)
        model = Model(inputs=[S,A],outputs=V)
        
        # Normal Optimization
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        
        return model, A, S    
