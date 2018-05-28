#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os, csv
from sklearn.model_selection import train_test_split

import tensorflow as tf
#import tensorlayer as tl

# Utilities :
def find_divisor(N) :
    div_lst = []
    for i in range(N) :
        if N % (i+1) == 0: div_lst.append(i+1)
    # Since last divisor is N itself, we keep the prior last one
    return div_lst

def error_rate(p, t):
    return np.mean(p != t)

############################
## Notes en fin de codes !!!
############################

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

class Neural_Network():
###-------------------------------------------------------------------------------
    def __init__(self, lr, N_={}, max_epoch=10, verbose=True, file_to_update = "", reduce_type = "sum", **kwargs) :
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
        
        self.lr = lr
        self.max_epoch = max_epoch
        self.file_to_update = file_to_update 
        
        if reduce_type =="sum" :
            self.reduce_type_fct = tf.reduce_sum
        
        if reduce_type =="mean" :   
             self.reduce_type_fct = tf.reduce_mean
             
        self.reduce_type = reduce_type
        
        self.savefile= "./session"
        self.sess = tf.InteractiveSession(config=config)
        
        
        self.batched = True if "bsz" in kwargs.keys() or "batch_sz" in kwargs.keys() else False
        
        try :
            self.color = kwargs["color"]
        except KeyError :
            self.color= 'purple'

        try :
            self.step = kwargs["step"]
        
        except KeyError :
            self.step = 10    
        
        if "BN" in kwargs.keys() :  
            self.BN = True
            print ("Batch Normalization method used")
        
        else :
            self.BN = False
            print ("Batch Normalization method NOT used")
        
        self.verbose= verbose 
        self.kwargs = kwargs

        ## Check on the keys : 
        keys = N_.keys()
        keys.remove("I"), keys.remove("O")
        
        sorted_keys = sorted(keys, key=lambda x: int(x[1:]))
        
        for j,k in enumerate(sorted_keys) :
            if int(k[1:]) != j+1 :
                raise KeyError ("Check N_ keys. One indice seems to be dumped: {}".format(keys))
        
        self.N_ = N_
        
###-------------------------------------------------------------------------------  
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
      
    def split_data(self, X, y, random_state=0, strat=False, standard_scale=False, shuffle=True, val=True):
#        https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation

#        Stratify option allows to have a loyal representation of the datasets (couples of data,target)
#        And allows a better training and can prevent from overfitting
        
#        Stratification is the process of rearranging the data as to ensure each fold is a good     
#        representative of the whole. For example in a binary classification problem where each class 
#        comprises 50% of the data, it is best to arrange the data such that in every fold, each class 
#        comprises around half the instances.

        if shuffle == True :
#        Inspired by : Sebastian Heinz
#        https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
            permute_indices = np.random.permutation(np.arange(len(y)))
            X = X[permute_indices]
            y = y[permute_indices]        
        
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
            print y.shape
        
        self.X, self.y = X, y
        
        if strat == True :
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
            self.strat_done = True
        else :
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
        
        X_train_stdd  =  X_train.std(axis=0)
        X_train_mean =  X_train.mean(axis=0)
        
        if standard_scale == True :
            # We have to normalize each dimension
            # We scale independently such that each dimension has 0 mean and 1 variance  
            
            # This is consistent with sklearn's StandardScale().fit(data)
            X_train_scaled = np.zeros_like(X_train)
            X_test_scaled = np.zeros_like(X_test)
            
            for i in range(X_train_mean.shape[0]) :
                X_train_scaled[:, i] = X_train[:, i] -  X_train_mean[i]
                X_test_scaled[:, i]  = X_test[:, i]  -  X_train_mean[i]
                
                if np.abs(X_train_stdd[i]) > 1e-12 :
                    X_train_scaled[:,i] /= X_train_stdd[i]
                    X_test_scaled[:,i] /= X_train_stdd[i]
            
            print ("Scaling done")
            print ("X_train_mean = \n{}\n X_train_stdd = \n{}".format(X_train_mean, X_train_stdd))
            X_train = X_train_scaled
            X_test = X_test_scaled
        
        if val == True :
            xtest_length = len(X_test)
            self.X_val = X_test[:int(xtest_length*0.2)]
            self.y_val = y_test[:int(xtest_length*0.2)]
                
            X_test = X_test[int(xtest_length*0.2):]
            y_test = y_test[int(xtest_length*0.2):]
            
        # ---        
        self.standard_scale = standard_scale
        self.X_test, self.y_test    =   X_test , y_test
        self.X_train, self.y_train  =   X_train, y_train
        self.X_train_mean, self.X_train_stdd =  X_train_mean, X_train_stdd

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
    
    def split_and_scale(self, X, y, scaler= "PCA", shuffle=True, val=False, random_state=0, n_components="mle", whiten=False) :
        
        if scaler == "standard" :
            self.split_data(X, y, shuffle=shuffle, strat=False,\
                            standard_scale=True, val=val, random_state = random_state)
        
        else : 
            if shuffle == True :
                # Source see above
                permute_indices = np.random.permutation(np.arange(len(y)))
                X = X[permute_indices]
                y = y[permute_indices]       

            if len(y.shape) == 1:
                y = y.reshape(-1,1)
            
            self.X, self.y = X, y
            
            xtr, xte, ytr, yte = train_test_split(X, y, random_state=random_state)        
                    
            if scaler=="minmax" :
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler().fit(xtr)
                
            if scaler == "robust" : 
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler().fit(xtr)
            
            if scaler == "PCA" :
                from sklearn.decomposition import PCA
    #            whiten : bool, optional (default False)
    #            When True (False by default) the `components_` vectors are multiplied
    #            by the square root of n_samples and then divided by the singular values
    #            to ensure uncorrelated outputs with unit component-wise variances.
                scaler = PCA(n_components=n_components, whiten=whiten).fit(xtr)
                
                print ("PCA allows another data representation.")
                print ("From %d inputs, N_ contain now %d inputs entries" %\
                        (    self.N_["I"],              scaler.n_components_))
                
                self.N_["I"] = scaler.n_components_
            
            self.X_train, self.y_train = scaler.transform(xtr), ytr
            self.X_test, self.y_test =  scaler.transform(xte), yte
            
        self.scaler = scaler
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------

    def scale_inputs(self, xs) :
        try :
            new_xs = np.asarray([xs[i] for i in range(len(xs))])
        
            if self.scaler == "standard" :
                for i, mean in enumerate(self.X_train_mean) :
                    new_xs[i] -= mean
                
                for i, stdd in enumerate(self.X_train_stdd) :
                    if np.abs(stdd) > 1e-8 :
                        new_xs[i] /= stdd
            
            else :
                new_xs = self.scaler.transform(new_xs.reshape(1,-1))
        
            return new_xs
            
        except AttributeError :
            print ("No scaling")
            return xs
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
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
###-------------------------------------------------------------------------------        
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
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------

    def layer_stacking_and_act(self, activation="relu"):
        Z = dict()
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu", "selu"]
        # to have leakyrelu upgrade tensorflow :  sudo pip install tensorflow==1.4.0-rc0 
        ### A wide variety of models are availables to compute the neuronal outputs.
        ### We chose that way of implementation : the main work is done only if str_act
        ### belongs to a list predefined that should be improved by time
        #####   Must read and cite articles or documentations related to activation fcts
        
        # cross entropy most likely to be used in classification wit sigmoid
        
        # We added the possibility to deal with internal covariate shifting (Better generalization)
        # This possibility is activated if "BN" is in kwargs
        
        for str_act, act in zip(act_func_lst, [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.leaky_relu, tf.nn.selu] ) :
            if str_act == activation :
                print ("fonction d'activation considérée : %s" %(activation))
                
                curr_zkey = "z1" # Should self a zlastkey
                # Computation of the first HL 
                m = tf.matmul(self.x,self.w_tf_d["w1"]) + self.b_tf_d["b1"]
                if self.BN == True :
                            # We modify the way m is defined
                    try :
                        decayBN = self.kwargs["decay"]
                    except KeyError :
                        decayBN = 0.999
                    m = tf.layers.batch_normalization(m, momentum=decayBN)

                        # Add the non linear activation
                Z[curr_zkey] = act(m)
                
                # Computation of other layers if existing        
                # Test if they exist and go ahead or pass to pred_model
                if len(self.N_.keys()) > 3 :
                    for i in range(2,len(self.w_tf_d)) : # i %d wi, We dont take wlast
                        curr_zkey, prev_zkey = "z%d" %(i), "z%d" %(i-1)
                        wkey, bkey = "w%d" %(i), "b%d" %(i)

                        m = tf.matmul(Z[prev_zkey], self.w_tf_d[wkey] )  + self.b_tf_d[bkey]
                        
                        if self.BN == True :
                            # We modify the way m is defined
                            try :
                                decayBN = self.kwargs["decay"]
                            except KeyError :
                                decayBN = 0.999
                            m = tf.layers.batch_normalization(m, momentum=decayBN)
                        
                        # Add the non linear activation
                        Z[curr_zkey] = act(m)

                # Compute the final layer to be fed in train or predict step 
                self.y_pred_model = tf.matmul(Z[curr_zkey], self.w_tf_d[self.wlastkey])\
                                    + self.b_tf_d[self.blastkey]
                
        if Z == {}:
            raise AttributeError ("\"{}\" activation function is unknown.\nActivation function must be one of {}".format(activation, act_func_lst))
        self.temp_weights = self.w_tf_d
        ### We constructed operations 
#        print("Z\'s construits")
        self.activation = activation
        self.Z = Z
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------

    def def_optimizer(self, train_mod) :
        """
        Define a training model. Use kwargs to specify particular training parameters.\n
        Documentation took from http://tflearn.org
        
        Args:
        -----
        train_mod   :   str     Define the string representation of the desired
                                optimization method.
                                Currently, has to be one of [\"RMS\", \"GD \", \"SGD\"].
                                Otherwise exits

        kwargs      :   dict    Dictionary that has to contain Optimizer specification.
                                Else default one will be used
        
        After execution of this method, the NN object will have the NN.train_op attribute 
        """
        considered_optimizer = {"RMS" : tf.train.RMSPropOptimizer,\
                                "Adam" : tf.train.AdamOptimizer,\
                                "GD": tf.train.GradientDescentOptimizer,\
                                "SGD": tf.train.GradientDescentOptimizer,\
                               
                                "Proximal_Adag": tf.train.ProximalAdagradOptimizer,\
                                "Proximal_Grad": tf.train.ProximalGradientDescentOptimizer\
                              }
        self.default = False
        keys = self.kwargs.keys()
        #--#
        if train_mod not in considered_optimizer.keys() : raise IndexError\
            ("{} Not recognized. Choose from {}".format(train_mod, considered_optimizer.keys()))
        #--#
        
        for k, v in zip(self.kwargs.keys(), self.kwargs.values()):
            print("kwargs[{}] = {}".format(k,v))
        
        if train_mod == "GD" or train_mod == "SGD":
            # Classical gradient descent. Insure to find global minimum, but may need a lot of time
            # GD or S(tochastic) GD are relying on the same tf optimizer : 
            self.train_op = considered_optimizer[train_mod](self.lr) 
        
        if train_mod=="RMS" :
        # Maintain a moving (discounted) average of the square of gradients. 
        # Divide gradient by the root of this average. 
        # Quickly catch model common feature representation, and allow to have the rare ones
            try :
                self.train_op = considered_optimizer[train_mod]\
                                   (self.lr, momentum=self.kwargs["momentum"],\
                                    decay=self.kwargs["decay"] )
            except KeyError:
                print("\x1b[1;37;43m\
                Seems like, some argument are missing in kwargs dict to design a RMSPROP optimizer\n\
                Use the default one instead with lr = {} though\
                \x1b[0m".format(self.lr))

                self.train_op = considered_optimizer[train_mod](self.lr)
                
                self.kwargs["momentum"]  =   0.0
                self.kwargs["decay"]     =   0.9    
                
                self.default = True
        
        if train_mod=="Adam" :
        # Classical Momentum added in RMSPROP algorithm
            if "beta1" in keys and "beta2" in keys:
                self.train_op =\
                        considered_optimizer[train_mod]\
                                (self.lr, beta1=self.kwargs["beta1"], beta2=self.kwargs["beta2"])
                                                        
            elif "beta1" in keys : 
                self.train_op =considered_optimizer[train_mod]\
                                (self.lr, beta1=self.kwargs["beta1"])
                self.kwargs["beta2"] = 0.99
            
            elif "beta2" in keys : 
                self.train_op = considered_optimizer[train_mod]\
                                (self.lr, beta2=self.kwargs["beta2"])
                self.kwargs["beta1"] = 0.9
                
            else :
                print("\x1b[1;37;43m\
                AdamOptimizer goes default lr = {}, beta1 = 0.9, beta2 = 0.99, epsilon = 10^(-8).\
                \x1b[0m".format(self.lr))
                
                self.train_op = considered_optimizer[train_mod](self.lr)  
                
                self.kwargs["beta1"] = 0.9
                self.kwargs["beta2"] = 0.99
                  
                self.default = True
        
        if train_mod == "Proximal_Grad": 
            if "l1reg" in keys and "l2reg" in keys :
                self.train_op = considered_optimizer[train_mod](self.lr,\
                                          l1_regularization_strength=self.kwargs["l1reg"],\
                                          l2_regularization_strength=self.kwargs["l2reg"])
            elif "l1reg" in keys :
                self.train_op = considered_optimizer[train_mod](self.lr,\
                                          l1_regularization_strength=self.kwargs["l1reg"])
                self.kwargs["l2reg"] = 0.0 
                
            elif "l2reg" in keys :
                self.train_op = considered_optimizer[train_mod](self.lr,\
                                          l2_regularization_strength=self.kwargs["l2reg"])
                self.kwargs["l1reg"] = 0.0
            else :  
                print("\x1b[1;37;43m\
                ProximalGradientDesent goes default lr = {}, l1 and l2 regularizer = 0.0\
                \x1b[0m".format(self.lr))
                
                self.train_op = considered_optimizer[train_mod](self.lr)
                
                self.kwargs["l1reg"], self.kwargs["l2reg"] = 0.0, 0.0
                
                self.default = True
                
        if train_mod == "Proximal_Adag": 
            if "l1reg" in keys and "l2reg" in keys :
                self.train_op = considered_optimizer[train_mod](self.lr,\
                                          l1_regularization_strength=self.kwargs["l1reg"],\
                                          l2_regularization_strength=self.kwargs["l2reg"])
                self.kwargs["initial_accumulator_value"] = 0.1
            elif "l1reg" in keys :
                self.train_op = considered_optimizer[train_mod](self.lr,\
                                          l1_regularization_strength=self.kwargs["l1reg"])
                self.kwargs["l2reg"] = 0.0 
                self.kwargs["initial_accumulator_value"] = 0.1
                
            elif "l2reg" in keys :
                self.train_op = considered_optimizer[train_mod](self.lr,\
                                          l2_regularization_strength=self.kwargs["l2reg"])
                self.kwargs["l1reg"] = 0.0
                self.kwargs["initial_accumulator_value"] = 0.1
            else :  
                print("\x1b[1;37;43m\
                ProximalGradientDesent goes default lr = {}, l1 and l2 regularizer = 0.0\
                \x1b[0m".format(self.lr))
                
                self.train_op = considered_optimizer[train_mod](self.lr)
                
                self.kwargs["l1reg"], self.kwargs["l2reg"] = 0.0, 0.0
                self.kwargs["initial_accumulator_value"] = 0.1
                
                self.default = True

        self.train_mod = train_mod
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------  
    
    def cost_computation(self, err_type, SL_type="regression") :
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
                             "Ridge": "below",\
                             "Lasso": "below",\
                             "Elastic": "below",\
                             "Custom": "below"}
            if err_type not in expected_loss :
                raise IndexError("{} n\'est pas dans les cas considérés. La fonction de cout doit appartenir à la liste {}".format(err_type, expected_loss.keys()))
            
            
            elif err_type == "Ridge":
                try :
                    ridge_param = self.kwargs["ridge_param"]
                except KeyError :
                    ridge_param = float(input("Give a Ridge parameter " ) )
                
                ridge_param = tf.constant(ridge_param)

                self.weight_sum = tf.placeholder(np.float32, (None))
                    
                self.loss = tf.expand_dims(\
                            tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t)),\
                            tf.multiply(ridge_param, self.reduce_type_fct(tf.square(self.weight_sum)))), 0)
            
            elif err_type == "Lasso":
                try :
                    lasso_param = self.kwargs["lasso_param"]
                except KeyError :
                    lasso_param = float(input("Give a Lasso parameter " ) )
                lasso_param = tf.constant(lasso_param)
                    
                self.weight_sum = tf.placeholder(np.float32, (None))
                
                self.loss = tf.expand_dims(\
                            tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t)),\
                            tf.multiply(lasso_param, self.reduce_type_fct(tf.abs(self.weight_sum)))), 0)
            
            elif err_type == "Elastic":
                try :
                    lasso_param = self.kwargs["lasso_param"]
                except KeyError :
                    lasso_param = float(input("Give a Lasso parameter " ) )
                    
                try :
                    ridge_param = self.kwargs["ridge_param"]
                except KeyError :
                    ridge_param = float(input("Give a Ridge parameter " ) )
                    
                lasso_param = tf.constant(lasso_param)
                ridge_param = tf.constant(ridge_param)
                
                self.weight_sum = tf.placeholder(np.float32, (None))
                    
                self.loss = tf.expand_dims(\
                            tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t)),\
                            tf.add(tf.multiply(lasso_param, self.reduce_type_fct(tf.abs(self.weight_sum))),\
                                   tf.multiply(ridge_param, self.reduce_type_fct(tf.square(self.weight_sum)))) ), 0)
            
            elif err_type == "Custom":
                try :
                    custom_param = self.kwargs["custom_param"]
                except KeyError :
                    custom_param = float(input("Give a custom parameter " ) )
            
                self.weight_sum = tf.placeholder(np.float32, (None))
                
                self.loss = tf.expand_dims(\
                            tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t)),\
                            tf.multiply(custom_param, self.reduce_type_fct(self.weight_sum))), 0)
                
            else :
                self.loss = self.reduce_type_fct(expected_loss[err_type](self.y_pred_model - self.t))
                
                print ("%s: The loss function will compute the averaged %s over all the errors" %(err_type, self.reduce_type))
                
#https://stackoverflow.com/questions/43822715/tensorflow-cost-function
#    Also tf.reduce_sum(cost) will do what you want, I think it is better to use tf.reduce_mean(). Here are a few reasons why:
#    you get consistent loss independent of your matrix size. On average you will get reduce_sum 4 times bigger for a two times bigger matrix
#    less chances you will get nan by overflowing

        self.err_type  = err_type 
        
###-------------------------------------------------------------------------------   
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
       
    def case_specification_recap(self) :
#    https://github.com/vsmolyakov/experiments_with_python/blob/master/chp03/tensorflow_optimizers.ipynb
        print("Récapitulatif sur la fonction de coût et la méthode de minimisation :\n")
        print("La méthode utilisée pour minimiser les erreurs entre les prédictions et les target est :{} -> {}\n".format(self.train_mod, self.train_op))
        print("La fonction de coût pour évaluer ces erreurs est {} -> {}".format(self.err_type, self.loss))
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------            
###-------------------------------------------------------------------------------

    def training_session(self, tol) :
#       Initialization ou ré-initialization ;)
        
        if self.err_type not in {"OLS", "AVL"} :
            self.training_coupled_regression(tol=tol)
            return "End of Ridge/Lasso or Elastic Network training"
        
        if "clip"  in self.kwargs.keys() :
            gradients, variables = zip(*self.train_op.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.minimize_loss = self.train_op.apply_gradients(zip(gradients, variables))
        else :
            self.minimize_loss = self.train_op.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        costs = []
        err, epoch, tol = 1., 0, tol
        
        if self.verbose == True :
#            if "label" in self.kwargs.keys() :
#                label = self.kwargs["label"]
#            else :
#                label = "%s %s %s %d size HL = %d " %\
#                (self.activation, self.train_mod, self.loss, self.bsz,len(self.N_.keys()) -2) 

            if plt.fignum_exists("Cost Evolution : loglog and lin") == False: 
                fig, axes = plt.subplots(1,2,figsize=(12,3), num="Cost Evolution : loglog and lin")
            subplot = plt.figure("Cost Evolution : loglog and lin")
            axes = subplot.axes
            fig = subplot
                
            axes[0].set_xlabel("#It")
            axes[0].xaxis.set_label_coords(-0.01, -0.06)
            axes[0].yaxis.set_label_position("left")
            axes[0].set_title("loglog")
            
            axes[1].yaxis.set_label_position("right")
            axes[1].set_title("lin")
            axes[1].set_ylabel("Errors")
            axes[1].yaxis.set_label_position("right")
            
                
        if self.batched == False :
#            with tf.Session() as sess:        
            while epoch <= self.max_epoch and err > tol:
                
                self.sess.run(self.minimize_loss,feed_dict={self.x : self.X_train, self.t : self.y_train})
                
                err = self.sess.run(self.loss, feed_dict={self.x : self.X_train,\
                                                          self.t : self.y_train})
                costs.append(err)
                if np.isnan(costs[-1]) : 
                    raise IOError("Warning, Epoch {}, lr = {}.. nan".format(epoch, self.lr))
                
                if epoch % self.step == 0 :
                    print("epoch {}/{}, cost = {}".format(epoch, self.max_epoch, err))
                    
                    if self.verbose == True :
                        axes[0].semilogy(epoch, costs[-1], marker='o', color=self.color)
                        axes[1].plot(epoch, costs[-1], marker='o', color=self.color)
                        
                        fig.tight_layout()
                        plt.pause(0.001)
                    
                epoch += 1

            print costs[-10:]
#            self.saver.save(sess, self.savefile)
        else :
            ff = open("just_a_test.txt", "w")
#            https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.py
            for epoch in range(self.max_epoch) :
#                print(self.kwargs["batch_sz"])
                n_batch, left = len(self.X_train) // self.kwargs["bsz"], len(self.X_train) % self.kwargs["bsz"]
#                print n_batch, left
                for b in range(n_batch) :
                    bsz = self.kwargs["bsz"] + 1 if left > 0 else self.kwargs["bsz"]
                    rand_index = np.random.choice(len(self.X_train), size=bsz)   

                    self.X_batch = self.X_train[rand_index]
                    self.y_batch = self.y_train[rand_index]
                    
                    left -=1
                    
                    if left <=0 : left = 0 
#                print self.X_batch.shape
#               
                    if self.BN == True :
                        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
                            self.sess.run(self.minimize_loss, feed_dict={self.x : self.X_batch,\
                                                                         self.t : self.y_batch})
                                            
                            err = self.sess.run(self.loss, feed_dict={self.x: self.X_train,\
                                                                      self.t: self.y_train})
                    
                    else :
#                X_batch = self.X_train[jj*batch_sz:(jj*batch_sz + batch_sz)]
#                y_batch = self.y_train[jj*batch_sz:(jj*batch_sz + batch_sz)]
                        self.sess.run(self.minimize_loss,feed_dict={self.x : self.X_batch,\
                                                                    self.t : self.y_batch})  
                                                                    
                        err = self.sess.run(self.loss, feed_dict={self.x: self.X_train,\
                                                                  self.t: self.y_train})
                        
                costs.append(err)
                
                if np.isnan(costs[-1]) : 
                    raise IOError("Warning, Epoch {}, lr = {}.. nan".format(epoch, self.lr))
                
                if epoch % self.step == 0 and epoch != 0:
                    print("epoch {}/{}, cost = {}".format(epoch, self.max_epoch, err))
                    if self.verbose == True :
                        axes[0].semilogy(epoch, costs[-1], marker='o', color=self.color)
                        axes[1].plot(epoch, costs[-1], marker='o', color=self.color)
                        
                        fig.tight_layout()
                        
                        plt.pause(0.001)
#                        print ("{} : \n{}".format(epoch, self.sess.run(self.w_tf_d[self.wlastkey])))
#                        ff.write("#"*20+"\n")
#                        ff.write("\nIt={}\nweights values\n{}\n".format(epoch, self.sess.run(self.w_tf_d)))
#                        ff.write("#"*20+"\n")
                if np.abs(costs[-1]) < 1e-6 :
                    print "Final Cost "
                    break
            print costs[-10:]
        ff.close()
        self.costs = costs

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###-------------------------------------------------------------------------------

    def training_coupled_regression(self, tol) :
        
        if "clip"  in self.kwargs.keys() :
            gradients, variables = zip(*self.train_op.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.minimize_loss = self.train_op.apply_gradients(zip(gradients, variables))
        else :
            self.minimize_loss = self.train_op.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        costs = []
        err, epoch, tol = 1., 0, tol
        
        if self.verbose == True :
#            if "label" in self.kwargs.keys() :
#                label = self.kwargs["label"]
#            else :
#                label = "%s %s %s %d size HL = %d " %\
#                (self.activation, self.train_mod, self.loss, self.bsz,len(self.N_.keys()) -2) 

            if plt.fignum_exists("Cost Evolution : loglog and lin") == False: 
                fig, axes = plt.subplots(1,2,figsize=(12,3), num="Cost Evolution : loglog and lin")
            subplot = plt.figure("Cost Evolution : loglog and lin")
            axes = subplot.axes
            fig = subplot
                
            axes[0].set_xlabel("#It")
            axes[0].xaxis.set_label_coords(-0.01, -0.06)
            axes[0].yaxis.set_label_position("left")
            axes[0].set_title("loglog")
            
            axes[1].yaxis.set_label_position("right")
            axes[1].set_title("lin")
            axes[1].set_ylabel("Errors")
            axes[1].yaxis.set_label_position("right")
            
        lst_key = self.N_.keys()
        lst_key.remove("I"); lst_key.remove("O")

        if self.batched == False :
#            with tf.Session() as sess:        
            while epoch <= self.max_epoch and err > tol:
                weight_sum = []

                for lk, _ in enumerate(lst_key) : 
                    curr_w = self.sess.run(self.w_tf_d.values()[lk])
                    for elt in curr_w.ravel() :
                        weight_sum.append(elt)
                
                weight_sum = np.array(weight_sum).reshape(-1,1)
                
                self.sess.run(self.minimize_loss,feed_dict={self.x : self.X_train,\
                                                            self.t : self.y_train,\
                                                            self.weight_sum : weight_sum})
            
                err = self.sess.run(self.loss, feed_dict={self.x : self.X_train,\
                                                      self.t : self.y_train,\
                                                      self.weight_sum : weight_sum})
                
                costs.append(err)
                if np.isnan(costs[-1]) : 
                    raise IOError("Warning, Epoch {}, lr = {}.. nan".format(epoch, self.lr))
                
                if epoch % self.step == 0 :
                    print("epoch {}/{}, cost = {}".format(epoch, self.max_epoch, err))
                    
                    if self.verbose == True :
                        axes[0].semilogy(epoch, costs[-1], marker='o', color=self.color)
                        axes[1].plot(epoch, costs[-1], marker='o', color=self.color)
                        
                        fig.tight_layout()
                        plt.pause(0.001)
                    
                epoch += 1

            print costs[-10:]
#            self.saver.save(sess, self.savefile)
        else :
            ff = open("just_a_test.txt", "w")
#            https://github.com/nfmcclure/tensorflow_cookbook/blob/master/03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/06_lasso_and_ridge_regression.py
            for epoch in range(self.max_epoch) :
#                print(self.kwargs["batch_sz"])
                n_batch, left = len(self.X_train) // self.kwargs["bsz"], len(self.X_train) % self.kwargs["bsz"]
#                print n_batch, left
                
                
                for b in range(n_batch) :
                    
                    weight_sum = []

                    for lk, _ in enumerate(lst_key) : 
                        curr_w = self.sess.run(self.w_tf_d.values()[lk])
                        for elt in curr_w.ravel() :
                            weight_sum.append(elt)
                    
                    weight_sum = np.array(weight_sum).reshape(-1,1)

                    bsz = self.kwargs["bsz"] + 1 if left > 0 else self.kwargs["bsz"]

                    rand_index = np.random.choice(len(self.X_train), size=bsz)   
                    self.X_batch = self.X_train[rand_index]
                    self.y_batch = self.y_train[rand_index]
                    
                    left -=1
                    
                    if left <=0 : left = 0 
#                print self.X_batch.shape
#               
                    if self.BN == True :
                        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
                            self.sess.run(self.minimize_loss, feed_dict={self.x : self.X_batch,\
                                                                         self.t : self.y_batch,\
                                                                         self.weight_sum : weight_sum})
                                            
                            err = self.sess.run(self.loss, feed_dict={self.x: self.X_train,\
                                                                      self.t: self.y_train,\
                                                                         self.weight_sum : weight_sum})
                    
                    else :
#                X_batch = self.X_train[jj*batch_sz:(jj*batch_sz + batch_sz)]
#                y_batch = self.y_train[jj*batch_sz:(jj*batch_sz + batch_sz)]
                        self.sess.run(self.minimize_loss,feed_dict={self.x : self.X_batch,\
                                                                    self.t : self.y_batch})  
                                                                    
                        err = self.sess.run(self.loss, feed_dict={self.x: self.X_train,\
                                                                  self.t: self.y_train})
                        
                costs.append(err)
                
                if np.isnan(costs[-1]) : 
                    raise IOError("Warning, Epoch {}, lr = {}.. nan".format(epoch, self.lr))
                
                if epoch % self.step == 0 and epoch != 0:
                    print("epoch {}/{}, cost = {}".format(epoch, self.max_epoch, err))
                    if self.verbose == True :
                        axes[0].semilogy(epoch, costs[-1], marker='o', color=self.color)
                        axes[1].plot(epoch, costs[-1], marker='o', color=self.color)
                        
                        fig.tight_layout()
                        
                        plt.pause(0.001)
#                        print ("{} : \n{}".format(epoch, self.sess.run(self.w_tf_d[self.wlastkey])))
#                        ff.write("#"*20+"\n")
#                        ff.write("\nIt={}\nweights values\n{}\n".format(epoch, self.sess.run(self.w_tf_d)))
#                        ff.write("#"*20+"\n")
                if np.abs(costs[-1]) < 1e-6 :
                    print "Final Cost "
                    break
            print costs[-10:]
        ff.close()
        self.costs = costs

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###-------------------------------------------------------------------------------        

    def predict(self, x_s):
        P = self.sess.run(self.y_pred_model, feed_dict={self.x: x_s})
        return P
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------

    def visualize_graph(self):
        writer = tf.summary.FileWriter('logs', self.sess.graph)
        writer.close()
        
        weights = tf.trainable_variables()
        print("Graph written. See tensorboard --logdir=\"logs\"")
        print self.sess.run(weights)
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
if __name__=="__main__":
    plt.ion()
    
    from sklearn.datasets import load_boston
    boston = load_boston()
    X, y = boston.data, boston.target

    gen_dict = lambda inputsize : \
               {"I"  : inputsize,\
               "N1" : 100,\
               "N2" : 100,\
               "N3" : 100,\
               "N4" : 100,\
               "O" :1}
               
    N_= gen_dict(X.shape[1])
    
#    lparams = [1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1]

    lparams = [1e-9, 1e-07, 1e-3]
    rparams = [1e-5, 1e-3, 1e-2]
#    lparams = [1e8, 1e7, 1e6, 100000, 10000, 1000, 100, 10, 1]
#    color = iter(cm.rainbow(np.linspace(0,1,np.shape(lparams)[0])))
#    color = iter(sns.diverging_palette(255, 30, l=80, n=len(lparams), center='dark'))
    color = iter(sns.color_palette("Set1", len(lparams)*len(rparams)))
    names = iter([str(i) + "_" + str(j)  for i in lparams for j in rparams])
    
    finalTrain_costs = []
    finalTest_costs = []
    
    elasticTrain_costs = []
    elasticTest_costs = []
    
    ll, rr = [], []
    
#    for l in lparams :
#        for r in rparams :
#            TF = Neural_Network(0.0005, N_=N_, reduce_type="sum", color=next(color), bsz=64, BN=True, verbose=True,\
#                                max_epoch=1000, clip=False, lasso_param = l, ridge_param = r)
#        
#            TF.train_and_split(X,y, scale=True, shuffle=True, val=False, strat=False)
#            
#            TF.tf_variables()
#            TF.layer_stacking_and_act("relu")
#            TF.def_optimizer("Adam")
#            TF.cost_computation("Elastic")
#            TF.case_specification_recap()
#            
#            TF.training_session(1e-3)
#            
#            elasticTrain_costs.append(TF.costs[-1][0])
#            
#            elasticTest_costs.append(TF.sess.run(TF.reduce_type_fct(tf.square(TF.predict(TF.X_test) - TF.y_test))))
#            
#            lab = next(names)
#            
#            plt.figure("Prediction")
#            plt.plot(TF.y_test, TF.y_test, label="Expected", color='k')
#            plt.plot(TF.y_test, TF.predict(TF.X_test), label="preds %s" % lab, color=TF.kwargs["color"], marker='o', linestyle="none", fillstyle='none')
#            plt.legend()
#            
#            plt.pause(0.1)
#            
#            TF.sess.close()
#            
#            ll.append(l)
#            rr.append(r)
#            
#    fig = plt.figure("3D Projection Path Elastic Net")
#    ax = fig.gca(projection='3d')
#    ax.plot(ll, rr, elasticTrain_costs, color='navy', label="Training cost evolution")
#    ax.plot(ll, rr, elasticTest_costs, color='darkred', label="Ontest set cost evolution")

#    ax.set_xlabel("Lasso param : $\lambda_L$")
#    ax.set_ylabel("Ridge param : $\lambda_R$")
#    ax.set_zlabel("Training costs")
#    #    ax=plt.gca()
#    #    ax.yaxis.set_label_coords(-0.08, 0.5)

#    #    ax.set_xlabel("Lasso param : $\lambda_L$")
#    #    ax.set_xlabel("Ridge param : $\lambda_R$")
#    #    ax.set_zlabel("Test costs")
#    #    ax=plt.gca()
#    #    ax.yaxis.set_label_coords(-0.08, 0.5)
#    ax.legend()

#plt.pause(0.1)
            
#    TF.error_computation("cross_entropy")
    
#    TF.optimisation()
    scaler_name = ["standard", "minmax", "robust", "PCA"]
    names = iter(scaler_name)
    color = iter(sns.color_palette("Set1", len(scaler_name)))
    
    for sn in scaler_name :
        TF = Neural_Network(0.0005, N_=N_, reduce_type="sum", color=next(color), bsz=64, BN=True, verbose=True,\
                            max_epoch=1000, clip=False)#, lasso_param = l, ridge_param = r)
    
        TF.split_and_scale(X,y, scaler=sn, shuffle=True, val=False)
    
        TF.tf_variables()
        TF.layer_stacking_and_act("relu")
        TF.def_optimizer("Adam")
        TF.cost_computation("Elastic")
        TF.case_specification_recap()
        
        TF.training_session(1e-3)
#    
        print (TF.costs[-1])
#    
        print(TF.sess.run(TF.reduce_type_fct(tf.square(TF.predict(TF.X_test) - TF.y_test))))
#    
        lab = next(names)
#    
        plt.figure("Prediction")
        plt.plot(TF.y_test, TF.y_test, label="Expected", color='k')
        plt.plot(TF.y_test, TF.predict(TF.X_test), label="preds %s" % lab, color=TF.kwargs["color"], marker='o', linestyle="none", fillstyle='none')
        plt.legend()

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
