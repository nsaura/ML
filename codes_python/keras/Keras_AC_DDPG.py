#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
#from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, concatenate, add, Lambda

from keras.layers.normalization import BatchNormalization as BN
#Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.

from keras.optimizers import Adam

import tensorflow as tf
import keras.backend as K

#HIDDEN1_UNITS = 80
#HIDDEN2_UNITS = 40

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
    """
    This class creates an actor network and a target actor network that will be modified.
    """
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, HIDDEN1_UNITS, HIDDEN2_UNITS):
        """
        During the initialisation, the first actor and a copy of it : the target actor are created.\
        In accordance with \"continuous control with deep reinforcement learning\" article, the target is to be modified\
        with a soft update using the sum of the original graph's weights times a discount factor tau with (1-tau) times the target's weights.  
        Several placeholders are created as well.  All the nodes of the graph are defined\
        It is then not necessary to go through the graph again to calculate the grads.
        
        The optimizer used is Adam.
        
        Constructor arguments :
        -------------------------
        sess        :   A tensorflow session
        state_size  :   The size of the states that are the input of the actor net
        action_size :   The size of the actions that are the output of the actor net
        BATCH_SIZE  :   The size of the batch considered here for the replay buffer memory
        TAU         :   The discounted factor to regulate how smooth is the update. Around 0.001
        LEARNING_RATE : Rate of correcting the weights in the optimizer
        """
        
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        # Prepare le LR decay
        self.LEARNING_RATE = tf.placeholder(tf.float32,shape=[]) 

        K.set_session(sess)

        #Now create the model
        #create actor network outs the model, weiths and inputs
        
        # First
        self.model, self.weights, self.state =\
                self.create_actor_network(state_size, action_size, HIDDEN1_UNITS, HIDDEN2_UNITS, name='Actor')
        
        # Target to be modified
        self.target_model, self.target_weights, self.target_state =\
                self.create_actor_network(state_size, action_size, HIDDEN1_UNITS, HIDDEN2_UNITS, name='Target_Actor')
        
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
        
        self.target_model.set_weights(actor_target_weights) # Equivalent to assign from tensorflow
    
    def create_actor_network(self, state_size,action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, name):
        S = Input(shape=[state_size], name=name+'_Input')
        h0 = Dense(HIDDEN1_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_Dense1')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_Dense2')(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_Dense3')(h1)
        V = Dense(action_dim ,activation=tanh, kernel_initializer = 'normal', name=name+'_Output')(h2)
        model = Model(inputs=S,outputs=V)
        model.summary()
        return model, model.trainable_weights, S

######################
######################
## Critique Network ##    
######################
######################

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, HIDDEN1_UNITS, HIDDEN2_UNITS):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state =\
                self.create_critic_network(state_size, action_size, HIDDEN1_UNITS, HIDDEN2_UNITS, name='Critic')
        self.target_model, self.target_action, self.target_state =\
                self.create_critic_network(state_size, action_size, HIDDEN1_UNITS, HIDDEN2_UNITS, name="Target_Critic") 

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
    
    def create_critic_network(self, state_size,action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, name):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_DenseS')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_DenseA2')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_DenseS2')(w1)
        h2 = concatenate([h1,a1])
        h3 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_DenseConca')(h2)
        h4 = Dense(HIDDEN2_UNITS, activation='selu', kernel_initializer = 'normal', name=name+'_DenseH4')(h3)
        V = Dense(1,activation='linear', name=name+'_Output')(h4)
        model = Model(inputs=[S,A],outputs=V)
        
        # Normal Optimization
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        
        return model, A, S    
        
# La suite n'est pas utilisée. On essaye deque dans le script Keras_AC_DDPG.py

# From https://github.com/hzwer/NIPS2017-LearningToRun/blob/master/baseline/rpm.py

# from collections import deque
# replay buffer per http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

#class rpm(object):
#    #replay memory
#    def __init__(self, buffer_size):
#        self.buffer_size = buffer_size
#        self.buffer = []
#        self.index = 0

#        import threading
#        self.lock = threading.Lock()

#    def add(self, obj):
#        self.lock.acquire()
#        if self.size() > self.buffer_size:

#            #trim
#            print('buffer size larger than set value, trimming...')
#            self.buffer = self.buffer[(self.size()-self.buffer_size):]

#        elif self.size() == self.buffer_size:
#            self.buffer[self.index] = obj
#            self.index += 1
#            self.index %= self.buffer_size

#        else:
#            self.buffer.append(obj)

#        self.lock.release()

#    def size(self):
#        return len(self.buffer)

#    def sample_batch(self, batch_size):
#        '''
#        batch_size specifies the number of experiences to add
#        to the batch. If the replay buffer has less than batch_size
#        elements, simply return all of the elements within the buffer.
#        Generally, you'll want to wait until the buffer has at least
#        batch_size elements before beginning to sample from it.
#        '''

#        if self.size() < batch_size:
#            batch = random.sample(self.buffer, self.size())
#        else:
#            batch = random.sample(self.buffer, batch_size)

#        item_count = len(batch[0])
#        res = []
#        for i in range(item_count):
#            # k = np.array([item[i] for item in batch])
#            k = np.stack((item[i] for item in batch),axis=0)
#            # if len(k.shape)==1: k = k.reshape(k.shape+(1,))
#            if len(k.shape)==1: k.shape+=(1,)
#            res.append(k)
#        return res
