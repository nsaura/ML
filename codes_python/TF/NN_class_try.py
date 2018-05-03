#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os, csv
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Utilities :
def find_divisor(N) :
    div_lst = []
    for i in range(N) :
        if N % (i+1) == 0: div_lst.append(i+1)
    # Since last divisor is N itself, we keep the prior last one
    return div_lst

def error_rate(p, t):
    return np.mean(p != t)


## Notes en fin de codes !!!

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

class Neural_Network():
###-------------------------------------------------------------------------------
    def __init__(self, lr, N_={}, max_epoch=10, verbose=False, file_to_update = "", **kwargs) :
        """
        On veut construire un réseau de neurones assez rapidement pour plusieurs cas.
        Cette classe prend les arguments suivants :
        Args:
        ------
        lr : float  --  Learning Rate (voir notes pour calibration)
        N_ : dict   --  Contient les noms des différentes couches.
                    Doit les première et dernière couches "I" and "O" resp. 
                    Les autres couches sont N1,...,Nn
        max_epoch : int --  Pour le learning, on impose une fin aux aller-retours du BackPropagation Algo.
                    Les poids seront peut être optimaux.
        """
        N_test = (N_ == "{}")
        if N_test == True :
            raise Exception("Merci de fournir un dictionnaire N_ de la forme : première clé \" \"I\" : N_features\"\
                            Puis les hidden layers sous la forme (pour la j-ème ) \" \"Nj\" : 100\"\
                            Enfin il faudra rajouter l'entrée \"\"O\" : 1 \"\n")

        self.wlastkey = "wlast"
        self.blastkey = "blast"
        
        print N_
        
        self.N_ = N_
        self.lr = lr
        self.max_epoch = max_epoch
        
        self.savefile= "./session"
        self.sess = tf.InteractiveSession(config=config)
        
        self.file_to_update = file_to_update 
        
        self.batched = True if "b_sz" in kwargs.keys() or "batch_sz" in kwargs.keys() else False
        
        try :
            self.color = kwargs["color"]
        except KeyError :
            self.color= 'orchid'

        try :
            self.step = kwargs["step"]
        
        except KeyError :
            self.step = 50    
            
        self.verbose= verbose 
        self.kwargs = kwargs
        
###-------------------------------------------------------------------------------        
    def train_and_split(self, X, y, random_state=0, strat=True, scale=False, shuffle=True):
#        Stratify option allows to have a loyal representation of the datasets (couples of data,target)
#        And allows a better training and can prevent from overfitting
        
#        Stratification is the process of rearranging the data as to ensure each fold is a good     
#        representative of the whole. For example in a binary classification problem where each class 
#        comprises 50% of the data, it is best to arrange the data such that in every fold, each class 
#        comprises around half the instances.

#        Stratification is generally a better scheme, both in terms of bias and variance, when 
#        compared to regular cross-validation.
#        https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation
        if shuffle == True :
#        Inspired by : Sebastian Heinz
#        https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
            permute_indices = np.random.permutation(np.arange(len(y)))
            X = X[permute_indices]
            y = y[permute_indices]        
        
        self.X, self.y = X, y
        
        strat_done = False
        if strat == True :
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
            strat_done = True
        else :
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
        
        X_train_std  =  X_train.std(axis=0)
        X_train_mean =  X_train.mean(axis=0)
#       Sometimes scaling datas leads to better generalization
        
        if scale == True :
            X_train_scaled = np.zeros_like(X_train)
            X_test_scaled = np.zeros_like(X_test)
            
            for i in range(X_train_mean.shape[0]) :
                X_train_scaled[:, i] = X_train[:, i] - X_train_mean[i]
                X_test_scaled[:, i] = X_test[:, i] - X_train_mean[i]
                
                if np.abs(X_train_std[i]) > 1e-12 :
                    X_train_scaled[:,i] /= X_train_std[i]
                    X_test_scaled[:,i] /= X_train_std[i]
            
            print ("Scaling done")
            print ("X_train_mean = \n{}\n X_train_std = \n{}".format(X_train_mean, X_train_std))
            X_train = X_train_scaled
            X_test = X_test_scaled
#            X_train[:,0] = (X_train[:,0]  - X_train_mean[0])
#            X_test[:,0]  = (X_test[:,0]   - X_train_mean[0])

#            if np.abs(X_std[0]) > 1e-12 :    
#                X_train[:,0]/= X_train_std[0]
#                X_test[:,0]  /= X_train_std[0]  

#            X_train[:,1] = (X_train[:,1]  - X_train_mean[1]) /   X_train_std[1]
#            X_test[:,1]  = (X_test[:,1]   - X_train_mean[1]) /   X_train_std[1]
        
#       We finish by "selfing" Training Set and Testing Set
        self.scale = scale
        self.X_test, self.y_test    =   X_test , y_test
        self.X_train, self.y_train  =   X_train, y_train
        self.X_train_mean, self.X_train_std =  X_train_mean, X_train_std
####-------------------------------------------------------------------------------
#    def mean_std_new_input(self, x_s):
#        for i in range(len(x_s)) :
#            x_s[i] -= self.X_train_mean[i]
#            
#            if np.abs(self.X_train_std[i]) > 1e-12 :
#                x_s[i] /= self.X_train_std[i]
#        
#        return x_s       
###-------------------------------------------------------------------------------
    def w_b_real_init(self):
        ### We use X_train.shape
        w_dict, biases_d = dict(), dict()
        Nraw_Xtrain, Ncol_Xtrain = self.X_train.shape

        self.N_["I"] = Ncol_Xtrain

        # We construct weight and biases dictionnary as we anticipate tensor dicts
        jj, err = 1, False # We skip N0 
        while jj <= len(self.N_.values()) and err == False : 
            wkey, bkey = "w%d_init" %(jj), "b%d_init" %(jj)

            prev_key = "N%d" %(jj-1) if jj != 1 else "I" 
            curr_key = "N%d" %(jj)
            try :
                w_dict[wkey] = np.random.randn(self.N_[prev_key],\
                        self.N_[curr_key]) / np.sqrt(self.N_[prev_key])
                biases_d[bkey] = np.zeros(self.N_[curr_key])
            
            except KeyError :
                w_dict[self.wlastkey] = np.random.randn(self.N_[prev_key],\
                        self.N_["O"]) / self.N_[prev_key]
                biases_d[self.blastkey] = np.zeros(self.N_["O"])
                err = True
                print err
                
            jj += 1

        self.w_dict = w_dict
        self.biases_d = biases_d
        self.Ncol_Xtrain = Ncol_Xtrain
###-------------------------------------------------------------------------------        
    def tf_variables(self):
        ### We want to create as many tf.tensors as we have weight and biases matrices 
            ### Those ones will be tf.Variables(weight.astype(np.float32))
        self.w_b_real_init()
           
        w_tf_d, b_tf_d = dict(), dict()
        wkeys = [i.split("_")[0] for i in self.w_dict.keys()]
        bkeys = [i.split("_")[0] for i in self.biases_d.keys()]

        for (kw,w, kb,b) in zip(wkeys, self.w_dict.values(), bkeys, self.biases_d.values()) :
            w_tf_d[kw] = tf.Variable(w.astype(np.float32))
            b_tf_d[kb] = tf.Variable(b.astype(np.float32))
        
        ### We also create x and t which are training data and target to train the model
            ### Those will be tf.placeholder 
        self.x = tf.placeholder(tf.float32, (None, self.Ncol_Xtrain) )
        self.t = tf.placeholder(tf.float32, (None, 1) )
        
        self.w_tf_d = w_tf_d
        self.b_tf_d = b_tf_d
###-------------------------------------------------------------------------------
    def feed_forward(self, activation="relu"):
        Z = dict()
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu", "selu"]
        # to have leakyrelu upgrade tensorflow :  sudo pip install tensorflow==1.4.0-rc0 
        ### A wide variety of models are availables to compute the neuronal outputs.
        ### We chose that way of implementation : the main work is done only if str_act
        ### belongs to a list predefined that should be improved by time
        #####   Must read and cite articles or documentations related to activation fcts
        
        # cross entropy most likely to be used in classification wit sigmoid
        for str_act, act in zip(act_func_lst, [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.leaky_relu, tf.nn.selu] ) :
            if str_act == activation :
                Z["z1"] = act( tf.matmul(self.x,self.w_tf_d["w1"]) + self.b_tf_d["b1"] )    
                print ("fonction d'activation considérée : %s" %(activation))
                for i in range(2,len(self.w_tf_d)) : # i %d wi, We dont take wlast
                    curr_zkey, prev_zkey = "z%d" %(i), "z%d" %(i-1)
                    wkey = "w%d" %(i)
                    bkey = "b%d" %(i)
                    Z[curr_zkey] = act( tf.matmul(Z[prev_zkey], self.w_tf_d[wkey] )  + self.b_tf_d[bkey] ) 
                self.y_pred_model = tf.matmul(Z[curr_zkey], self.w_tf_d[self.wlastkey]) + self.b_tf_d[self.blastkey]
                
        if Z == {}:
            raise AttributeError ("\"{}\" activation function is unknown.\nActivation function must be one of {}".format(activation, act_func_lst))
        self.temp_weights = self.w_tf_d
        ### We constructed operations 
#        print("Z\'s construits")
        self.activation = activation
        self.Z = Z
###-------------------------------------------------------------------------------      
    def def_training(self, train_mod) :
        """
        Define a training model. Use kwargs to specify particular training parameters.\n
        Documentation took from http://tflearn.org
        
        Args:
        -----
        train_mod   :   str     Define the string representation of the desired optimization method.
                                Currently, has to be one of [\"RMS\", \"GD \", \"SGD\"]. Otherwise exits
        kwargs      :   dict    Dictionary that has to contain Optimizer specification. Else default one will be used
        
        After execution of this method, the NN object will have the NN.train_op attribute 
        """
        considered_optimizer = {"RMS" : tf.train.RMSPropOptimizer,\
                                "Adam" : tf.train.AdamOptimizer,\
                                "Nadam" : tf.keras.optimizers.Nadam(),\
                                "GD": tf.train.GradientDescentOptimizer,\
                                "SGD": tf.train.GradientDescentOptimizer\
                              }
        self.default = False
        if train_mod not in considered_optimizer.keys() :
            raise IndexError("{} n\'est pas dans les cas considérés. l\'argument train mod doit faire partie de la liste {}".format(train_mod, considered_optimizer.keys()))
        
        print("Optimiseur choisi : {}".format(train_mod))
        
        for k, v in zip(self.kwargs.keys(), self.kwargs.values()):
            print("kwargs[{}] = {}".format(k,v))
        
        if train_mod == "GD" or train_mod == "SGD":
            # Classical gradient descent. Insure to find global minimum, but may need a lot of time
            # GD or S(tochastic) GD are relying on the same tf optimizer : 
            self.train_op = considered_optimizer[train_mod](self.lr) 
        
        if train_mod=="RMS" :
        # Maintain a moving (discounted) average of the square of gradients. 
        # Divide gradient by the root of this average. 
        # Quickly catch model common feature representation, and allow to have rare feature representation
            try :
                self.train_op = considered_optimizer[train_mod](\
                                                               self.lr,\
                                                               momentum=self.kwargs["momentum"],\
                                                               decay=self.kwargs["decay"]\
                                                                )
            except KeyError:
                print("\x1b[1;37;43mSeems like, some argument are missing in kwargs dict to design a RMSPROP optimizer\n\
                Use the default one instead with lr = {} though\x1b[0m".format(self.lr))
                self.train_op = tf.train.RMSPropOptimizer(self.lr)
                
                self.kwargs["momentum"]  =   0.0
                self.kwargs["decay"]     =   0.9    
                
                self.default = True
        
        if train_mod=="Adam" :
        # Classical Momentum added in RMSPROP algorithm
            if "beta1" in self.kwargs.keys() and "beta2" in self.kwargs.keys():
                self.train_op = considered_optimizer[train_mod](\
                                                               self.lr,\
                                                               beta1=self.kwargs["beta1"],\
                                                               beta2=self.kwargs["beta2"],\
                                                              )
            elif "beta1" in self.kwargs.keys() : 
                self.train_op = considered_optimizer[train_mod](\
                                                               self.lr,\
                                                               beta1=self.kwargs["beta1"]\
                                                              )
                self.kwargs["beta2"] = 0.99
            
            elif "beta2" in self.kwargs.keys() : 
                self.train_op = considered_optimizer[train_mod](\
                                                               self.lr,\
                                                               beta2=self.kwargs["beta2"]\
                                                              )
                self.kwargs["beta1"] = 0.9
                
            else :
                print("\x1b[1;37;43mAdamOptimizer goes default beta1 = 0.9, beta2 = 0.99, epsilon = 10^(-8). Though lr is specified = {} instead of dafault 0.001\x1b[0m".format(self.lr))
                self.train_op = tf.train.AdamOptimizer(self.lr)  
                
                self.kwargs["beta1"] = 0.9
                self.kwargs["beta2"] = 0.99
                  
                self.default = True
        
                    
        # Selfing the optimizer with the wanted tuning 
        self.train_mod = train_mod
###-------------------------------------------------------------------------------      
    def cost_computation(self, err_type, SL_type="regression", reduce_type = "sum") :
#        CLASSIFICATION pbs : cross entropy most likely to be used in  with sigmoid : softmax_cross_entropy_with_logits
#        REGRESSION pbs     : L2 regularisation -> OLS  :   tf.reduce_sum(tf.square(y_pred - targets))
#                             L1 regression     -> AVL  :   tf.reduce_sum(tf.abs(y_pred - targets))
        
        if SL_type == "classification" :
            self.loss = tf.reduce_sum(\
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred_model, label=self.t)\
                                     )
            
        else :
            expected_loss = {"OLS" : tf.square,\
                             "AVL" : tf.abs,\
                             "VSGD": "below",\
                             "Ridge": "below",\
                             "Lasso": "below"}
            if err_type not in expected_loss :
                raise IndexError("{} n\'est pas dans les cas considérés. La fonction de cout doit appartenir à la liste {}".format(err_type, expected_loss.keys()))
            
            if err_type == "VSGD":
                print ("Voir plus loin")
                self.loss = 1
            
            elif err_type == "Ridge" or err_type == "ridge":
                if "ridge_param" not in self.kwargs.keys() :
                    ridge_param = float(input("Give a Ridge parameter " ) )
                    ridge_param = tf.constant(ridge_param)
                    if reduce_type == "mean" :
                        ridge_loss = tf.reduce_mean(tf.square(self.x))
                        self.loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(self.y_pred_model - self.t)), tf.multiply(ridge_param, ridge_loss)), 0)
                    
                    if reduce_type == "sum" :
                        ridge_loss = tf.reduce_sum(tf.square(self.x))
                        self.loss = tf.expand_dims(tf.add(tf.reduce_sum(tf.square(self.y_pred_model - self.t)), tf.multiply(ridge_param, ridge_loss)), 0)

            else :
                if reduce_type == "mean" :
                    self.loss = tf.reduce_mean(expected_loss[err_type](self.y_pred_model - self.t))
                    print ("%s: The loss function will compute the averaged sum over all the errors" % err_type)
                elif reduce_type == "sum" :
                    self.loss = tf.reduce_sum(expected_loss[err_type](self.y_pred_model - self.t))
                    print ("%s: The loss function will compute sum over all the errors" % err_type)
                else :
                    raise AttributeError("Define a reduce_type between sum and mean")
                
#https://stackoverflow.com/questions/43822715/tensorflow-cost-function
#    Also tf.reduce_sum(cost) will do what you want, I think it is better to use tf.reduce_mean(). Here are a few reasons why:
#    you get consistent loss independent of your matrix size. On average you will get reduce_sum 4 times bigger for a two times bigger matrix
#    less chances you will get nan by overflowing

        self.reduce_type = reduce_type
        self.err_type  = err_type 
###-------------------------------------------------------------------------------          
    def def_optimization(self, verbose = False):
#    https://github.com/vsmolyakov/experiments_with_python/blob/master/chp03/tensorflow_optimizers.ipynb
        if verbose == True :
            print("Récapitulatif sur la fonction de coût et la méthode de minimisation :\n")
            print("La méthode utilisée pour minimiser les erreurs entre les prédictions et les target est :{} -> {}\n".format(self.train_mod, self.train_op))
            print("La fonction de coût pour évaluer ces erreurs est {} -> {}".format(self.err_type, self.loss))
        
        if self.err_type != "VSGD" :
            self.minimize_loss = self.train_op.minimize(self.loss)
###-------------------------------------------------------------------------------
    def training_session(self, tol, verbose = False) :
#       Initialization ou ré-initialization ;)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        if "VSGD" in self.kwargs.keys():
            if self.err_type == "VSGD" or self.err_type == "Vanilla" :
                sum_weight = 0
                ww = self.sess.run(self.w_tf_d)
                reg = float(input("Enter Vanila regularization coefficient "))
                if self.reduce_type == "mean" :
                    meansq = tf.reduce_mean(tf.square(self.y_pred_model - self.t))
                    for w in ww.iteritems() :
                        sum_weight += np.mean(w[1])
                    self.loss = tf.expand_dims(tf.add(meansq, reg*0.5*sum_weight), 0)
            
                if self.reduce_type == "sum":
                    meansq = tf.reduce_sum(tf.square(self.y_pred_model - self.t))
                    for w in ww.iteritems() :
                        sum_weight += np.sum(w[1])
                    self.loss = tf.expand_dims(tf.add(meansq, reg*0.5*sum_weight), 0)

            self.minimize_loss = self.train_op.minimize(self.loss)

        costs = []
        err, epoch, tol = 1., 0, tol
        
        if verbose == True :
            plt.figure("Cost Evolution")
        
        if self.batched == False :
#            with tf.Session() as sess:        
            while epoch <= self.max_epoch and err > tol:
                self.sess.run(self.minimize_loss,feed_dict={self.x : self.X_train, self.t : self.y_train})
                
                err = self.sess.run(self.loss, feed_dict={self.x : self.X_train,\
                                                          self.t : self.y_train})
                costs.append(err)
                if np.isnan(costs[-1]) : raise IOError("Warning, Epoch {}, lr = {}.. nan"\
                                                    .format(epoch, self.lr))
                if epoch % self.step == 0 :
                    print("epoch {}/{}, cost = {}".format(epoch, self.max_epoch, err))
                    if verbose == True :
                        plt.plot(epoch, costs[-1], marker='o', color=self.color, linestyle='--')
                        plt.pause(0.001)
#                    print ("{} : \n{}".format(epoch, self.sess.run(self.w_tf_d[self.wlastkey])))
                    
                epoch += 1

            print costs[-10:]
#            self.saver.save(sess, self.savefile)
        else :
#            https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.py
            for epoch in range(self.max_epoch) :
#                print(self.kwargs["batch_sz"])
                rand_index = np.random.choice(len(self.X_train), size=self.kwargs["b_sz"])   

                self.X_batch = self.X_train[rand_index]
                self.y_batch = self.y_train[rand_index]
#                print self.X_batch.shape
#                
                
#                X_batch = self.X_train[jj*batch_sz:(jj*batch_sz + batch_sz)]
#                y_batch = self.y_train[jj*batch_sz:(jj*batch_sz + batch_sz)]
                self.sess.run(self.minimize_loss,feed_dict={self.x : self.X_batch, self.t : self.y_batch})  
                err = self.sess.run(self.loss, feed_dict={self.x: self.X_train,\
                                                          self.t: self.y_train})
                    
                costs.append(err)
                
                if np.isnan(costs[-1]) : 
                    raise IOError("Warning, Epoch {}, lr = {}.. nan".format(epoch, self.lr))
                
                if epoch % self.step == 0 and epoch != 0:
                    print("epoch {}/{}, cost = {}".format(epoch, self.max_epoch, err))
                    if verbose == True :
                        plt.plot(epoch, costs[-1], marker='o', color=self.color, linestyle='--')
                        plt.pause(0.001)
                
                if np.abs(costs[-1]) < 1e-4 :
                    print "Final Cost "
                    continue
            print costs[-10:]
        self.costs = costs
###-------------------------------------------------------------------------------
    def predict(self, x_s):
        P = self.sess.run(self.y_pred_model, feed_dict={self.x: x_s})
        return P
###-------------------------------------------------------------------------------
    def visualize_graph(self):
        writer = tf.summary.FileWriter('logs', self.sess.graph)
        writer.close()
        
        weights = tf.trainable_variables()
        print("Graph written. See tensorboard --logdir=\"logs\"")
        print self.sess.run(weights)
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
if __name__=="__main__":
    
    TF = Neural_Network(0.0004, model="", pathfile="test.csv", N_hidden_layer=3, layers_sizes = [2, 100, 100, 200, 1])
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    TF.train_and_split(cancer.data, cancer.target)
    TF.build_graph()
    TF.tf_variables()
    TF.feed_forward("relu")
    TF.error_computation("cross_entropy")
    
    TF.optimisation()


##-------------------------------------------------- NOTES --------------------------------------------------##


## Notes intéressantes de Methods of Model Based Process Control :

# "The back propagation-based learning algorithm [..] varies only the weights of the neural network to achieve the desired mapping
# To overcome dependence of the learning process on the initial settings and for further improvement of the mapping accuracy, we use a combination of backpropagation and a genetic algorithm. The key idea os to vary the properties of each neurom in the net simultaneously with the adaptation of the weigths
# The processing properties of a neuron are determined by its activation function. 

# Exemple OLS + L1 regression :
# From https://stackoverflow.com/questions/36706379/how-to-exactly-add-l1-regularisation-to-tensorflow-error-function

#import tensorflow as tf

#total_loss = meansq #or other loss calcuation
#l1_regularizer = tf.contrib.layers.l1_regularizer(
#   scale=0.005, scope=None)
#   
#weights = tf.trainable_variables() # all vars of your graph
#regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

#regularized_loss = total_loss + regularization_penalty # this loss needs to be minimized
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(regularized_loss)


#How to set adaptive learning rate for GradientDescentOptimizer?
#https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer
#learning_rate = tf.placeholder(tf.float32, shape=[])
## ...
#train_step = tf.train.GradientDescentOptimizer(
#    learning_rate=learning_rate).minimize(mse)

#sess = tf.Session()

## Feed different values for learning rate to each training step.
#sess.run(train_step, feed_dict={learning_rate: 0.1})
#sess.run(train_step, feed_dict={learning_rate: 0.1})
#sess.run(train_step, feed_dict={learning_rate: 0.01})
#sess.run(train_step, feed_dict={learning_rate: 0.01})
