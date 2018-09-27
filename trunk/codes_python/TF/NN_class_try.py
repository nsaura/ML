#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os

from sklearn.model_selection import train_test_split


from datetime import datetime
import custom_swish_function as csf

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

csf = reload(csf)

import tensorflow as tf

# Utilities :
def find_divisor(N) :
    div_lst = []
    for i in range(N) :
        if N % (i+1) == 0: div_lst.append(i+1)
    # Since last divisor is N itself, we keep the prior last one
    return div_lst


#############################
# Notes en fin de codes !!! #
#############################

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

class Neural_Network():
###-------------------------------------------------------------------------------
    def __init__(self, lr, N_, scaler, max_epoch=10, verbose=True, graph_name = "./default.ckpt", reduce_type = "sum", **kwargs) :
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
        try :
            tf.reset_default_graph()
        except :
            print ("This time the graph won't be deleted")
            pass
        
        N_test = (N_ == "{}")
        if N_test == True :
            raise Exception("Merci de fournir un dictionnaire N_ de la forme : première clé \" \"I\" : N_features\"\
                            Puis les hidden layers sous la forme (pour la j-ème ) \" \"Nj\" : 100\"\
                            Enfin il faudra rajouter l'entrée \"\"O\" : 1 \"\n")

        self.wlastkey = "wlast"
        self.blastkey = "blast"
        
        print ("dict_layers = {}".format(N_))
        
        self.lr = lr
        self.max_epoch = max_epoch
        self.scaler_name = scaler        
        self.graph_name = graph_name
        
        if reduce_type =="sum" :
            self.reduce_type_fct = tf.reduce_sum
        
        if reduce_type =="mean" :   
             self.reduce_type_fct = tf.reduce_mean
             
        self.reduce_type = reduce_type
                
        self.savefile= "./session"
        self.sess = tf.InteractiveSession(config=config)
        

        try :
            self.color = kwargs["color"]
        except KeyError :
            self.color= 'purple'
            kwargs["color"] = self.color

        try :
            self.step = kwargs["step"]
        
        except KeyError :
            self.step = 10    
        
        self.batched = True if "bsz" in kwargs.keys() or "batch_sz" in kwargs.keys() else False

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
        print (keys, type(keys))
        keys.remove("I")
        keys.remove("O")
        
        sorted_keys = sorted(keys, key=lambda x: int(x[1:]))
        
        for j,k in enumerate(sorted_keys) :
            if int(k[1:]) != j+1 :
                raise KeyError ("Check N_ keys. One indice seems to be dumped: {}".format(keys))
        
        self.N_ = N_
        self.exception = 0
        
###-------------------------------------------------------------------------------  
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
    
    def split_and_scale(self, X, y, shuffle=True, val=False, random_state=0, n_components="mle", whiten=False) :
        
        X, y = np.copy(X), np.copy(y)
        print ("Scaling used : %s " % self.scaler_name)
        if self.scaler_name == "None" :
            self.split_data(X, y, shuffle=shuffle, strat=False,\
                            standard_scale=False, val=val, random_state = random_state)
            scaler = self.scaler_name
            
        elif self.scaler_name == "Standard" :
            self.split_data(X, y, shuffle=shuffle, strat=False,\
                            standard_scale=True, val=val, random_state = random_state)
            scaler = self.scaler_name
        
        else : #We split and preparing data with different scalers
            self.standard_scale = self.scaler_name
            
            if shuffle == True :
                # Source see above
                permute_indices = np.random.permutation(np.arange(len(y)))
                X = X[permute_indices]
                y = y[permute_indices]
                self.permute_indices =  permute_indices       

            if len(y.shape) == 1:
                y = y.reshape(-1,1)
            
            self.X, self.y = X, y
            
            xtr, xte, ytr, yte = train_test_split(X, y, random_state=random_state)        
                    
            if self.scaler_name=="MinMax" :
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler().fit(xtr)
                
            if self.scaler_name == "Robust" : 
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler().fit(xtr)
            
            if self.scaler_name == "PCA" :
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
        
        print ("Standard scale = {}".format(self.standard_scale))
        print ("scaler = %s" % self.scaler_name)
        
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
        X, y = np.copy(X), np.copy(y)
        
        if shuffle == True :
#        Inspired by : Sebastian Heinz
#        https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877
            permute_indices = np.random.permutation(np.arange(len(y)))
            X = X[permute_indices]
            y = y[permute_indices]     
            self.permute_indices =  permute_indices   
        
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        
        self.X, self.y = X, y
        
        if strat == True :
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
            self.strat_done = True
        else :
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
        
        X_train_mean =  X_train.mean(axis=0)
        X_train_stdd  =  X_train.std(axis=0, ddof=1)
        
        if standard_scale == True :
            # We have to normalize each dimension
            # We scale independently such that each dimension has 0 mean and 1 variance  
            
            # This is consistent with sklearn's StandardScale().fit(data)
            X_train_scaled = np.zeros_like(X_train)
            X_test_scaled = np.zeros_like(X_test)
            
            for i, mean in enumerate(X_train_mean) :
                X_train_scaled[:, i] = X_train[:, i] -  mean
                X_test_scaled[:, i]  = X_test[:, i]  -  mean
                
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
    
    def scale_inputs(self, xs) :
        """
        Doc to be done
        """
        new_xs = np.copy(xs)
        input_shape = np.shape(new_xs)
        
        if  len(input_shape) == 2 and np.shape(new_xs)[0] > 1 :
            print ("Use of scale_inputs method only for one line element")
            print ("Use nn_obj.predict(arr, rescale_tab=True) to rescale an array")
            sys.exit()
            
        if np.shape(new_xs)[0] < 1 :
            sys.exit("Null Argument in scale_inputs call")
        
        try :
            if self.scaler_name == "Standard" :
                
                for i, (mean, std) in enumerate(zip(self.X_train_mean, self.X_train_stdd)):
#                    print new_xs
#                    print new_xs.shape
#                    print mean
                    
                    new_xs[i] = new_xs[i] - mean

                    if np.abs(std) > 1e-5 :
                        new_xs[i] = new_xs[i] / std
            
            else :
                new_xs = self.scaler.transform(new_xs.reshape(1,-1))
        
            return new_xs.reshape(1,-1)
            
        except AttributeError :
            self.exception += 1
            
            if self.exception == 1 :
                print("\x1b[1;37;41mNo scaling\x1b[0m")
            
            else :
                pass
        
            return new_xs.reshape(1,-1)
        
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
                w_dict[wkey]=np.random.randn(self.N_[prev_key],self.N_[curr_key])/np.sqrt(self.N_[prev_key])
                biases_d[bkey] = np.zeros(self.N_[curr_key])
            
            except KeyError :
                w_dict[self.wlastkey] = np.random.randn(self.N_[prev_key],\
                        self.N_["O"]) / self.N_[prev_key]
                biases_d[self.blastkey] = np.zeros(self.N_["O"])
                err = True
#                print err
                
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
            w_tf_d[kw] = tf.Variable(w.astype(np.float32), name=kw)
            b_tf_d[kb] = tf.Variable(b.astype(np.float32), name=kb)
        
        ### We also create x and t which are training data and target to train the model
            ### Those will be tf.placeholder 
        self.x = tf.placeholder(tf.float32, (None, self.Ncol_Xtrain), name="inputs"  )
        self.t = tf.placeholder(tf.float32, (None, self.N_["O"]), name="output" )
        
        self.w_tf_d = w_tf_d
        self.b_tf_d = b_tf_d
    
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------

    def layer_stacking_and_act(self, activation="relu"):
        Z = dict()
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu", "selu", "swish"]
        # to have leakyrelu upgrade tensorflow :  sudo pip install tensorflow==1.4.0-rc0 
        ### A wide variety of models are availables to compute the neuronal outputs.
        ### We chose that way of implementation : the main work is done only if str_act
        ### belongs to a list predefined that should be improved by time
        #####   Must read and cite articles or documentations related to activation fcts
        
        # cross entropy most likely to be used in classification wit sigmoid
        
        # We added the possibility to deal with internal covariate shifting (Better generalization)
        # This possibility is activated if "BN" is in kwargs
        
        for str_act, act in zip(act_func_lst, [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.leaky_relu, tf.nn.selu, csf.tf_swish] ) :
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
                self.train_op = considered_optimizer[train_mod]\
                                 (self.lr, beta1=self.kwargs["beta1"], beta2=self.kwargs["beta2"])
                                                        
            elif "beta1" in keys : 
                self.train_op = considered_optimizer[train_mod]\
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
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred_model, label=self.t))
        
        else :
            classic_loss = {"OLS" : tf.square,
                             "AVL" : tf.abs} 
            advanced_loss=["MSEGrad", "Ridge", "Lasso", "Elastic", "Custom"]
            
            other_loss   = ["FVC", "CBS"]
            
            if err_type in other_loss :
                self.advanced=False
            
            if err_type in advanced_loss :
                self.advanced = True

            key_to_check = advanced_loss + classic_loss.keys() + other_loss
            
            if err_type not in key_to_check :
                raise IndexError("{}\'s not in the list {}".format(err_type, key_to_check))
                
            elif err_type == "MSEGrad" :
                self.jac = tf.placeholder(tf.float32, (None), name="grad_y_pred_x")
                self.loss = self.reduce_type_fct(tf.add(tf.square(self.y_pred_model - self.t), tf.multiply(1e-5, self.jac)))
                
            elif err_type == "Ridge":
                self.Elastic_cost(r=0)
            
            elif err_type == "Lasso":
                self.Elastic_cost(r=1)
            
            elif err_type == "Elastic":
                self.Elastic_cost()
                
            elif err_type == "Custom":
                self.Custom_cost()
            
            elif err_type == "CBS" :
                self.CBS()
            
            elif err_type == "Full_vector_cost" or err_type == "FVC" :
                self.FVC()
            
            else :
                self.loss = self.reduce_type_fct(classic_loss[err_type](self.y_pred_model - self.t))
                self.advanced = False
                
                print ("%s: The loss function will compute the averaged %s over all the errors" %(err_type, self.reduce_type))
                
#https://stackoverflow.com/questions/43822715/tensorflow-cost-function
#    Also tf.reduce_sum(cost) will do what you want, I think it is better to use tf.reduce_mean(). Here are a few reasons why:
#    you get consistent loss independent of your matrix size. On average you will get reduce_sum 4 times bigger for a two times bigger matrix
#    less chances you will get nan by overflowing
        self.err_type  = err_type 
        self.SL_type = SL_type
###-------------------------------------------------------------------------------   
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
       
    def case_specification_recap(self) :
#    https://github.com/vsmolyakov/experiments_with_python/blob/master/chp03/tensorflow_optimizers.ipynb
        print("Récapitulatif sur la fonction de coût et la méthode de minimisation :\n")
        print("La méthode utilisée pour minimiser les erreurs entre les prédictions et les target est :{} -> {}\n".format(self.train_mod, self.train_op))
        print("La fonction de coût pour évaluer ces erreurs est {} -> {}".format(self.err_type, self.loss))
        
        
        log_path = os.path.abspath("./logs")
        now = datetime.utcnow().strftime("%Y_%m_%d_%Hh%m_%S")
        bsz = self.kwargs["bsz"] if self.batched == True else len(self.X_train)

        dir_name = os.path.join(log_path, now + "_{}_{}_mepoch{}_bsz{}".format(self.train_mod, self.activation, self.max_epoch, bsz))
        
        log_dir = "{}/run-{}".format(dir_name, now)
        
        self.log_dir = log_dir
        self.dir_name = dir_name
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------            
###-------------------------------------------------------------------------------

    def training_phase(self, tol, early_stop=False) :
        
        if "clip"  in self.kwargs.keys() and self.kwargs["clip"] == True :
            gradients, variables = zip(*self.train_op.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.minimize_loss = self.train_op.apply_gradients(zip(gradients, variables))
            print ("Clip !")
            
        else :
            self.minimize_loss = self.train_op.minimize(self.loss)        
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        ##  End of the construction Phase 
                
        saver = tf.train.Saver() # Saver
        case_summary = tf.summary.scalar(self.err_type, self.loss) # We write the loss whose name is err_type

        #We use log_dir (case_specification) with current graph        
        file_writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph()) 
        
        costs = []
        score_test = []
        
        epoch = 0
        
        # Executing Phase
        if self.err_type == "MSEGrad" :
            self.grads_inputs = tf.norm(tf.gradients(self.y_pred_model, [self.x])[0])
        
        tf.get_default_graph().finalize()
        
        if early_stop == True :
            tol = 0.93
            err = 0.0
            condition = lambda epoch, err : (epoch <= self.max_epoch and err < tol)
        
        else :
            err = 1000.
            condition = lambda epoch, err : (epoch <= self.max_epoch and err > tol)
        
        if self.verbose == True :
            # Prepare the figures the errors will be plotting in.
            fig, axes = self.config_plot()
            
        while condition(epoch, err):
            if self.batched== True :
                    n_batch, left = len(self.X_train) // self.kwargs["bsz"], len(self.X_train) % self.kwargs["bsz"]
                    err, feeding = self.batched_training(n_batch, left)
                    curr_effective_epoch = n_batch * epoch 
                    final_effective_epoch = n_batch * self.max_epoch
                    
                    legend_to_plots = "Train error (with batch_size = %d)" % (self.kwargs["bsz"])
             
            else :
                err, feeding = self.non_batched_training()
                curr_effective_epoch = epoch 
                final_effective_epoch = self.max_epoch
                
                legend_to_plots = "Train error (without batch)"
                
            if early_stop == True :
                y_pred = self.predict(self.X_train, rescale_tab=False)
                score_test.append(self.score(y_pred, self.y_train))
            costs.append(err)

            if np.isnan(costs[-1]) : 
                raise IOError("Warning, (effective) Epoch {}, lr = {}.. nan".format(curr_effective_epochepoch, self.lr))
            
            if epoch % self.step == 0 :
                if epoch % (2*self.step) :
                    summary_str = case_summary.eval(feed_dict=feeding)
                    file_writer.add_summary(summary_str, curr_effective_epoch)
                
                if self.verbose == True :
                    axes[0].semilogy(epoch, costs[-1], marker='o', color=self.color, linestyle="None")
                    axes[1].plot(epoch, costs[-1], marker='o', color=self.color, linestyle="None")
                    
                    fig.tight_layout()
                    plt.pause(0.001)
                
                print("(effective) epoch {}/{}, cost = {}".format(curr_effective_epoch, final_effective_epoch, err))
            
            for a in axes :
                a.legend([legend_to_plots])
                
            if np.abs(costs[-1]) < 1e-6 :
                print ("Final Cost ".format(costs[-1]))
                break
            
            if early_stop == True :
                err = score_test[-1]
                
            epoch += 1
        
        print ("Ten last costs :\n{} ".format(costs[-10:]))
        print ("Last epoch : %d" % epoch)
        
        saver_path = saver.save(self.sess, self.graph_name)
        file_writer.close()
        
        self.saver_path = saver_path
        self.file_writer = file_writer
        self.case_summary = case_summary
        
        self.costs = costs
        
        if plt.fignum_exists("Cost Evolution : loglog and lin") :
            leg_0 = axes[0].get_legend()
            leg_1 = axes[1].get_legend()
            
            leg_0.legendHandles[-1].set_color(self.color)
            leg_1.legendHandles[-1].set_color(self.color)
            
            fig.tight_layout()
            
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------

    def batched_training(self, n_batch, left): 
        lst_key = self.N_.keys()
        lst_key.remove("I"); lst_key.remove("O")
        err_b = 0
        for b in range(n_batch) :
            
            bsz = self.kwargs["bsz"] + 1 if left > 0 else self.kwargs["bsz"]

            rand_index = np.random.choice(len(self.X_train), size=bsz)   
            self.X_batch = self.X_train[rand_index]
            self.y_batch = self.y_train[rand_index]
            
            left -=1
            if left <=0 : left = 0 
            
            if self.advanced == True and self.err_type != "MSEGrad" :
                weight_sum = []

                for lk, _ in enumerate(lst_key) : 
                    curr_w = self.sess.run(self.w_tf_d.values()[lk])
                    for elt in curr_w.ravel() :
                        weight_sum.append(elt)
                
                weight_sum = np.array(weight_sum).reshape(-1,1)
                feeding = {self.x : self.X_batch, self.t : self.y_batch, self.weight_sum : weight_sum}
            
            elif self.err_type=="MSEGrad" : 
                feeding = {self.x : self.X_batch, self.t : self.y_batch, self.jac:self.grads_inputs}      
                
            else : 
                feeding = {self.x : self.X_batch, self.t : self.y_batch}
                
            if self.BN == True :
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
                    if self.err_type=="MSEGrad" :
                        feeding = {self.x : self.X_batch, self.t : self.y_batch,
                           self.jac : self.sess.run(self.grads_inputs, feed_dict={self.x : self.X_batch})}
                        
                    # Tuning the parameters (weights, biais)
                    self.sess.run(self.minimize_loss, feed_dict=feeding)
                    
                    # Assessing the cost with the updated parameters             
                    err_b += self.sess.run(self.loss, feed_dict=feeding)
        err = err_b
        return err, feeding
            
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###------------------------------------------------------------------------------- 

    def non_batched_training(self): 
        lst_key = self.N_.keys()
        lst_key.remove("I"); lst_key.remove("O")
            
        if self.advanced == True and self.err_type != "MSEGrad":
            weight_sum = []

            for lk, _ in enumerate(lst_key) : 
                curr_w = self.sess.run(self.w_tf_d.values()[lk])
                for elt in curr_w.ravel() :
                    weight_sum.append(elt)
            
            weight_sum = np.array(weight_sum).reshape(-1,1)
            feeding = {self.x : self.X_train, self.t : self.y_train, self.weight_sum : weight_sum}
            
        else  :
            feeding = {self.x : self.X_train, self.t : self.y_train}
        
        if self.err_type=="MSEGrad" :
            feeding = {self.x : self.X_train, self.t : self.y_train,
                       self.jac : self.sess.run(self.grads_inputs, feed_dict={self.x : self.X_train})}
        
        # Tuning the parameters (weights, biais)
        self.sess.run(self.minimize_loss, feed_dict=feeding)
                    
        # Assessing the cost with the updated parameters                
        err = self.sess.run(self.loss, feed_dict=feeding)

        return err, feeding

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###-------------------------------------------------------------------------------

    def config_plot(self): 
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

        return fig, axes

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###------------------------------------------------------------------------------- 

    def Elastic_cost(self, r='None') :
        # Pseudo code : 
        # J = tf.square(y_pred - target) + r*alpha * sum(|weights|) + (1-r)*0.5*alpha * sum(tf.square(weights))
        
        # si r = 0 ---> Ridge
        # si r = 1 ---> Lasso 
        if r == 'None' :
        
            if "r_parameter" in self.kwargs.keys() :
               r = self.kwargs["r_parameter"]
            else :  
                r = -50
                while np.abs(r) > 1 :
                    r = float(input("Enter a value of r within 0 (all Ridge) and 1 (all lasso)" ))
        
        alpha = 1e-7 if "alpha" not in self.kwargs.keys() else self.kwargs["alpha"]
                    
        if np.abs(r) < 1e-8 :
            print ("r = %f --> Ridge function selected" % r)
        
        if np.abs(r-1) < 1e-8 :
            print ("r = %f --> Lasso function selected" % r)
        
        self.weight_sum = tf.placeholder(np.float32, (None), name="weight_sum")
            
        self.loss = tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t)),\
                    tf.add(tf.multiply(r*alpha, self.reduce_type_fct(tf.abs(self.weight_sum))),\
                           tf.multiply((1.-r)*0.5*alpha, self.reduce_type_fct(tf.square(self.weight_sum)))))
    
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###------------------------------------------------------------------------------- 

    def Custom_cost(self): 
        custom_choice = ["custom_param", "Custom_param", "custom_parameter"]
        inside = [c in self.kwargs.keys() for c in custom_choice]
        
        if True in inside :
            custom_param = self.kwargs[custom_choice[inside.index[True]]]
        
        else :
            print ("Custom Param set to {}".format(1e-4))
            custom_param = 1e-4
            self.kwargs["custom_param"] = custom_param
       
        self.loss = tf.expand_dims(\
                    tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t)),\
                    tf.multiply(custom_param, self.reduce_type_fct(self.x))), 0)

###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###------------------------------------------------------------------------------- 

    def FVC(self): 
        FVC_param = 10. if "FVC_param" not in self.kwargs.keys() else self.kwargs["FVC_param"]

        self.loss = tf.add(self.reduce_type_fct(tf.square(self.y_pred_model - self.t )),
                    tf.multiply(FVC_param, self.reduce_type_fct( tf.abs( tf.square(self.y_pred_model) - tf.square(self.t) ) ))
                          ) 
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------    
###-------------------------------------------------------------------------------
    
    def predict(self, xs, rescale_tab=False):
        arg = np.copy(xs)
        
        if len(arg.shape) == 1 :
            arg = arg.reshape(1,-1)
        
        if rescale_tab == True :        
            if self.scaler_name != "None" :
                for i, mean in enumerate(self.X_train_mean) :
                    arg[:, i] -= mean            
                for i, std in enumerate(self.X_train_stdd) :
                    if np.abs(std) > 1e-8 :
                        arg[:, i] /= std            
        
        P = self.sess.run(self.y_pred_model, feed_dict={self.x: arg})
        return P
        
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
    
    def score (self, y_pred, y_true, rescale_tab=False) :
        y_pred, y_true = np.copy(y_pred), np.copy(y_true)

        if self.SL_type == "classification" :
            score = np.mean(y_pred != y_true)
            self.mean_score = score
            
        if self.SL_type == "regression" :
            y_true_mean = np.mean(y_true)
            
            ratio = sum((y_true - y_pred)**2) / sum((y_true - y_true_mean)**2)
            score = 1. - ratio
            self.r2score = score
        
        return score 
                    
###-------------------------------------------------------------------------------             
###-------------------------------------------------------------------------------
###-------------------------------------------------------------------------------
    def visualize_graph(self):
#        writer = tf.summary.FileWriter('logs', self.sess.graph)
#        writer.close()
        
#        weights = tf.trainable_variables()
        print("Graph written. See tensorboard --logdir={}".format(self.log_dir))
#        print self.sess.run(weights)
        
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
               "N2" : 10,\
               "N3" : 10,\
               "N4" : 10,\
               "O" :1}
               
    N_= gen_dict(X.shape[1])
    
    scaler_name = ["Standard", "MinMax", "Robust", "PCA"]
    names = iter(scaler_name)
    
    TF = Neural_Network(0.0005, N_=N_, scaler = "Standard", reduce_type="sum", color="blue", verbose=True, max_epoch=1000, clip=False, r_parameter=0.5)#, bsz=64, BN=True)#, lasso_param = l, ridge_param = r)

    TF.split_and_scale(X,y,shuffle=True, val=False)

    TF.tf_variables()
    TF.layer_stacking_and_act("relu")
    TF.def_optimizer("Adam")
    TF.cost_computation("OLS")
    TF.case_specification_recap()
    
    TF.training_phase(1e-3)
#    
    plt.figure("Prediction")
    plt.plot(TF.y_test, TF.y_test, label="Expected", color='k')
    plt.plot(TF.y_test, TF.predict(TF.X_test), label="preds Standard", color=TF.kwargs["color"], marker='o', linestyle="none", fillstyle='none')
    plt.legend()

##-------------------------------------------------- NOTES --------------------------------------------------##

## Notes intéressantes de Methods of Model Based Process Control :

### Quelques notes sur les gradients : https://stackoverflow.com/questions/41822308/how-tf-gradients-work-in-tensorflow

#dc_dw, dc_db = tf.gradients(cost, [W, b])

# The tf.gradients() function allows you to compute the symbolic gradient of one tensor with respect to one or more other tensors—including variables.

# Full exemple from : https://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable

#data = tf.placeholder(tf.float32)
#var = tf.Variable(...)              # Must be a tf.float32 or tf.float64 variable.
#loss = some_function_of(var, data)  # some_function_of() returns a `Tensor`.

#var_grad = tf.gradients(loss, [var])[0]
#sess = tf.Session()
#var_grad_val = sess.run(var_grad, feed_dict={data: ...})

#https://books.google.fr/books?id=bRpYDgAAQBAJ&pg=PT656&lpg=PT656&dq=you+can+call+optimizer.compute_gradients()+with+trainable+variable&source=bl&ots=h3VGTaqIcM&sig=2q7x5ABYbzWvuQ8H8pT_CMycZtE&hl=fr&sa=X&ved=0ahUKEwjsjNm868HbAhVJVRQKHRg0A6MQ6AEIVDAE#v=onepage&q=you%20can%20call%20optimizer.compute_gradients()%20with%20trainable%20variable&f=false
#https://www.tensorflow.org/api_docs/python/tf/gradients


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
