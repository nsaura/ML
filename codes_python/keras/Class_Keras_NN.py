#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os
import os.path as osp
from sklearn.model_selection import train_test_split

import keras 
import tensorflow as tf

import argparse

plt.ion()
#------------------------------------------------------------------------------- 
def parser() :
    parser = argparse.ArgumentParser(description='Parser for NN with keras')
    parser.add_argument('--metrics', '-metrics', action='store', nargs='+', type=str, dest="metrics",\
                        default=['mean_squared_error', 'mean_absolute_error'],\
                        help="Define metrics calculated at the end of an epoch")
                        
    parser.add_argument('--optimizer', '-opti', action='store', type=str, default="Adam",\
                        dest="opti", help='Define an optimizer. Default %(default)s\n')
    parser.add_argument('--activation', '-act', action='store', type=str, default="relu",\
                        dest="act", help='Define an activation function if the same for all nodes\n')
    parser.add_argument('--loss_function', '-loss', action='store', type=str, default='mse',\
                        dest="loss", help='Define a loss function to be minimized. Default to mse')
    
    parser.add_argument('--max_epoch', '-maxepoch', action='store', type=int, default=1000,\
                        dest="maxepoch", help='Define the number of epochs in the training. Default 1000')
    
    parser.add_argument('--decay', '-decay', action='store', type=float, default=0.0,\
                        dest="decay", help="Define the learning rate decay. Default 0.0")
    parser.add_argument('--schedule_decay', '-schedule_decay', action='store', type=float, default=0.004,\
                        dest="schedule_decay", help="Define the learning rate schedule_decay (Nadam Algo). Default 0.004")
    
    parser.add_argument('--beta1', '-beta1', action='store', type=float, default=0.9,\
                        dest="beta1", help='Define beta_1 for Adam, Adamax or Nadam algo. Default to 0')
    parser.add_argument('--beta2', '-beta2', action='store', type=float, default=0.999,\
                        dest="beta2", help='Define beta_2 for Adam, Adamax or Nadam algo. Default to 0')
    
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.0,\
                        dest="momentum", help='Define momentum for SGD algo. Default to 0')
    parser.add_argument('--learning_rate', '-lr', action='store', type=float, default=0.001,\
                        dest="lr", help='Define the learning_rate for every algo. Default to 0')
    
    parser.add_argument('--batch_size', '-bsz', action='store', type=int, default=0,\
                        dest="bsz", help='Define the size of Batch, none is 0')
    
    parser.add_argument('--rho', '-rho', action='store', type=float, default=0.0,\
                        dest="rho", help='Define rho for RMSprop algo. Default to 0')
    
    parser.add_argument('--nesterov_bool', '-nest', action='store', type=bool, default=False,\
                        dest='nest', help="Specify if you want to use Nesterov Momentum (NAG) (SGD Algo)")
#    parser.add_argument('--amsgrad_bool', '-amsgrad', action='store', type=bool, default=False,\
#                        dest='amsgrad', help="Specify if you want to use amsgrad (Adam Algo)")
    return parser.parse_args()
#------------------------------------------------------------------------------- 

class K_Neural_Network () :
#------------------------------------------------------------------------------- 
    def __init__(self, dict_layers, opti, loss, metrics=['mean_squared_error', 'mean_absolute_error'], max_epoch=10, verbose=False, **kwargs):
        self.dict_layers = dict_layers
        self.max_epoch = max_epoch
        
        self.non_uniform_act = True if "non_uniform_act" in kwargs.keys() else False
        self.batched = True if kwargs["batch_size"] != 0 else False
        
        try :
            self.color = kwargs["color"]
        except KeyError :
            self.color= 'orchid'

        try :
            self.step = kwargs["step"]
        
        except KeyError :
            self.step = 50    
            
        self.verbose = verbose
        self.kwargs = kwargs
        
        print kwargs
        
        self.model = keras.models.Sequential() #pour l'instant
        
        self.conv_str_to_acti = {"relu" : keras.layers.activations.relu,\
                                 "selu" : keras.layers.activations.selu,\
                                 "linear" : keras.layers.activations.linear\
                                 }
#                   defined as a class "leakyrelu" : keras.layers.advanced_activations.LeakyReLU\
        
        self.conv_str_to_opti = {"RMSprop" : keras.optimizers.RMSprop,\
                                 "Adamax" : keras.optimizers.Adagrad,\
                                 "Nadam" : keras.optimizers.Nadam,\
                                 "Adam" : keras.optimizers.Adam,\
                                 "SGD" : keras.optimizers.SGD\
                                }

        self.conv_str_to_metr = {"mse" : keras.metrics.mse,\
                                 "mae" : keras.metrics.mae\
                                }   

        if opti not in self.conv_str_to_opti.keys() :
            raise KeyError("opti selected: {} is not in {}".format(opti, self.conv_str_to_opti.keys()))
                                
        self.conv_str_to_loss = {"mae" : keras.losses.MAE,\
                                 "logcosh": keras.losses.logcosh,\
                                 "mse" : keras.losses.mse\
                                }
        self.opti, self.loss = opti, loss
        
        self.keras_opti = self.conv_str_to_opti[opti]
        self.keras_loss = self.conv_str_to_loss[loss]
        
#        self.metrics = [self.conv_str_to_metr[i] for i in metrics]
        self.metrics = metrics
#-------------------------------------------------------------------------------  
    def build(self, act=None):
        layers = {}
        
        dict_key = self.dict_layers.keys()
        dict_key.remove("I")
        dict_key.remove("O")
         
        sorted_keys = sorted(dict_key, key = lambda x: int(x[1:]))
        
        layers["I"] = keras.layers.Input(shape=(self.dict_layers["I"],))
        
        if self.non_uniform_act == True :
            
            for j, k in enumerate(sorted_keys) : # k = N1; k = N2 etc
                
                if j == 0 :
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           activation= self.conv_str_to_acti[self.dict_layers[k][1]],\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer='zeros',\
                                           input_dim=self.dict_layers["I"]\
                                          ))
                else :                    
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           activation= self.conv_str_to_acti[self.dict_layers[k][1]],\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer='zeros',\
                                           name='Dense-%s' %(k)\
                                          ))
            self.model.add(keras.layers.Dense(self.dict_layers["O"][0],\
                                     activation=self.conv_str_to_acti[self.dict_layers["O"][1]],\
                                     kernel_initializer='random_uniform',\
                                     bias_initializer ='zeros',\
                                     name='Output-Layer'\
                                  ))
        else :
            for j, k in enumerate(sorted_keys) : # k = N1; k = N2 etc
                if j == 0 :
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           input_dim=self.dict_layers["I"],\
                                           activation=keras.layers.activations.relu,\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer ='zeros'\
                                          ))
                                          
                else :
                    self.model.add(keras.layers.Dense(self.dict_layers[k][0],\
                                           activation=keras.layers.activations.relu,\
                                           kernel_initializer='random_uniform',\
                                           bias_initializer ='zeros',\
                                           name='Dense-%s' %(k)\
                                          ))
               #Output line
            self.model.add(keras.layers.Dense(self.dict_layers["O"][0],\
                                    activation=keras.layers.activations.relu,\
                                    kernel_initializer='random_uniform',\
                                    bias_initializer ='zeros',\
                                    name='Output-Layer'\
                                 ))
        self.layers = layers
#-------------------------------------------------------------------------------
    def summary(self):
        print("-"*20)
        print("Input dictionnary :\n \x1b[1;37;43m{}\x1b[0m".format(self.dict_layers))
        print ("Optimizer choosen: {}".format(self.opti))
        
        # There might be some problem in the following action.
        # Have a look here to solve them https://github.com/XifengGuo/CapsNet-Keras/issues/7
        # Basically sudo apt-get install graphviz and update pydot with pip as said in the link
        print(self.model.summary())
        
        from keras.utils.vis_utils import plot_model
        
        filename, cpt = "model.png", 0
        if osp.exists("./models") == False :
            os.mkdir("./models")
        filename = osp.join("models", filename)
        
        while osp.exists(filename)==True :
            if cpt == 0:
                filename = osp.splitext(filename)[0] + "-%d" %(cpt+1) + ".png"
            else :
                filename = osp.splitext(filename)[0][:filename.index("-")+1] + "%d" %(cpt+1) + ".png"
            cpt +=1
        
        print ("Current Graphe can be seen by taping in a terminal :") 
        print ("\"eog {} \"".format(osp.abspath(filename)))
         
        plot_model(k_nn.model, to_file=filename)
#-------------------------------------------------------------------------------  
    def train_and_split(self, X, y, random_state=0, strat=False, scale=False, shuffle=True):
        if shuffle == True :
            # Inspired by : Sebastian Heinz se :
            # Medium : a simple deep-learning model for stock price prediction using tensorflow
            permute_indices = np.random.permutation(np.arange(len(y)))
            X = X[permute_indices]
            y = y[permute_indices] 

        if strat == True :
            if np.size(np.shape(y)) == 1 :
                xxyys = train_test_split(X, y, stratify=y,\
                        random_state=random_state)
            if np.size(np.shape(y)) == 2 :
                xxyys = train_test_split(X, y, stratify=y.reshape(-1),\
                        random_state=random_state)
        else :
            xxyys = train_test_split(X, y, random_state=random_state)
        
        X_train, X_test = xxyys[0], xxyys[1]
        y_train, y_test = xxyys[2], xxyys[3]
        
        X_train_std  =  X_train.std(axis=0)
        X_train_mean =  X_train.mean(axis=0)
        
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
        
        self.scale = scale
        self.X_train_mean = X_train_mean
        self.X_train_std  = X_train_std
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
#-------------------------------------------------------------------------------
    def compile(self, save=True, name="non-trained-model-1.h5"):
        self.summary()
        
        if self.opti == "SGD" :
            optimizer = self.keras_opti(lr = self.kwargs["lr"],\
                                        decay = self.kwargs["decay"],\
                                        momentum = self.kwargs["momentum"],\
                                        nesterov = self.kwargs["nesterov"]\
                                       )
        
        if self.opti == "RMSprop" :
            optimizer = self.keras_opti(lr = self.kwargs["lr"],\
                                        rho = self.kwargs["rho"],\
                                        decay = self.kwargs["decay"]\
                                       )

        if self.opti in ["Adam"] :
            optimizer = self.keras_opti(lr = self.kwargs["lr"],\
                                        beta_1 = self.kwargs["beta1"],\
                                        beta_2 = self.kwargs["beta2"],\
                                        decay  = self.kwargs["decay"]\
                                       )
        
        if self.opti in ["Adamax"] :
            optimizer = self.keras_opti(lr = self.kwargs["lr"],\
                                        beta_1 = self.kwargs["beta1"],\
                                        beta_2 = self.kwargs["beta2"],\
                                        decay  = self.kwargs["decay"]\
                                       )
        
        if self.opti == "Nadam" :
            optimizer = self.keras_opti(lr = self.kwargs["lr"],\
                                        beta_1 = self.kwargs["beta1"],\
                                        beta_2 = self.kwargs["beta2"],\
                                        schedule_decay = self.kwargs["schedule_decay"]\
                                       )
        #### Faire de même avec RMS et SGD
        ### Faire des try except pour lancer les commandes par defaut si non precise dans le kwargs 
        
        self.model.compile(loss=self.keras_loss, optimizer=optimizer, metrics=self.metrics)

        if save == True :
            self.model.save(name)
#-------------------------------------------------------------------------------
    def fit(self):           
        bsz = self.kwargs["batch_size"] if self.batched == True else len(self.X_train)
        self.fit = self.model.fit(self.X_train, self.y_train, epochs=self.max_epoch, batch_size=bsz)
        
        lenm = len(self.metrics)
        color = iter(cm.magma_r(lenm))
        
        for i in self.metrics :
            c = next(color)
            plt.plot(self.fit.history['%s' % i], label="%s" % i)
        
        plt.legend()
        self.prediction = self.model.predict
#-------------------------------------------------------------------------------
if __name__ == '__main__' : 
    par = parser()
    
    def defined_optimizer(parser):
        default_value = {"SGD"   : {"lr"    :   0.01,\
                                    "momentum" : 0.0,\
                                    "decay" :   0.0,\
                                    "nesterov" : False},\
                         
                         "RMSprop":{"lr"    :   0.001,\
                                    "rho"   :   0.9,\
                                    "decay" :   0.0},\
                         
                         "Adam"  : {"lr"    :   0.001,\
                                    "decay" :   0.0,\
                                    "beta1" :   0.9,\
                                    "beta2" :   0.999},\
                         
                         "Adamax": {"lr"    :   0.002,\
                                    "beta1" :   0.9,\
                                    "beta2" :   0.999,\
                                    "decay" :   0.0},\
                         
                         "Nadam" : {"lr"    :   0.002,\
                                    "beta1" :   0.9,\
                                    "beta2" :   0.999,\
                                    "schedule_decay" : 0.004}\
                        }
                                    
        dico_pars = par.__dict__
        kwargs = {}

        print ("-"*20)
        
        if par.opti=="SGD" :
            print ("\x1b[1;37;44mSGD Choosen\x1b[0m")
            print ("Parameters are :")
            print ("\t \tlr = %f\n\
                    momentum = %f\n\
                    decay = %f\n\
                    Nesterov (boolean) = %r" % (par.lr, par.momentum, par.decay, par.nest)
                  )
            if bool(input("Do you want to change them (1/0)? " )) :
                kwargs["lr"] = float(input("lr (default = %f): " % (default_value[par.opti]["lr"])))
                kwargs["momentum"] = float(input("momentum (default = %f): " % (default_value[par.opti]["momentum"])))
                kwargs["decay"] = float(input("decay (default = %f): " % (default_value[par.opti]["decay"])))
                kwargs["nesterov"] = bool(input("nest (default = %f): " % (default_value[par.opti]["nesterov"])))
                
                print ("\x1b[1;37;43mSGD: lr = %f, momentum = %f, decay = %f, nesterov = %r\x1b[0m"\
                            % (kwargs["lr"], kwargs["momentum"], kwargs["decay"], kwargs["nesterov"]))

            else :
                kwargs["lr"] = par.lr
                kwargs["momentum"] = par.momentum
                kwargs["decay"] = par.decay
                kwargs["nesterov"] = par.nest
            
        if par.opti == "RMSprop" :
            print ("\x1b[1;37;44mRMSprop Choosen\x1b[0m")
            print ("Parameters are :")
            print ("\t \tlr = %f\n\
                    rho = %f\n\
                    decay = %f" % (par.lr, par.rho, par.decay)
                  )
            
            if bool(input("Do you want to change them (1/0)? " )) :
                kwargs["lr"] = float(input("lr (default = %f): " % (default_value[par.opti]["lr"]) ))
                kwargs["rho"] = float(input("rho (default = %f): " % (default_value[par.opti]["rho"]) ))
                kwargs["decay"] = float(input("decay (default = %f): " % (default_value[par.opti]["decay"])))
                
                print ("\x1b[1;37;43mRMSprop: lr = %f, rho = %f, decay = %f\x1b[0m"\
                            % (kwargs["lr"], kwargs["rho"], kwargs["decay"]))

            else :
                kwargs["lr"] = par.lr
                kwargs["rho"] = par.rho
                kwargs["decay"] = par.decay

        
        if par.opti == "Adam" :
            print ("\x1b[1;37;44mAdam Choosen\x1b[0m")
            print ("Parameters are :")
            print ("\t \tlr = %f\n\
                    beta1 = %f\n\
                    beta2 = %f\n\
                    decay = %f\n" % (par.lr, par.beta1, par.beta2, par.decay,)
                  )
            if bool(input("Do you want to change them (1/0)? " )) :
                kwargs["lr"] = float(input("lr (default = %f): " % (default_value[par.opti]["lr"])))
                kwargs["beta1"] = float(input("beta1 (default = %f): " % (default_value[par.opti]["beta1"])))
                kwargs["beta2"] = float(input("beta2 (default = %f): " % (default_value[par.opti]["beta2"])))
                kwargs["decay"] = float(input("decay (default = %f): " % (default_value[par.opti]["decay"])))

                print ("\x1b[1;37;43m%s: lr = %f, beta1 = %f, beta2 = %f, decay = %f\x1b[0m"\
                            % (par.opti, kwargs["lr"], kwargs["beta1"],\
                               kwargs["beta2"], kwargs["decay"])  
                      )
            else :
                kwargs["lr"] = par.lr
                kwargs["beta1"] = par.beta1
                kwargs["beta2"] = par.beta2
                kwargs["decay"] = par.decay
        
        if par.opti == "Adamax" :
            print ("\x1b[1;37;44mAdamax Choosen\x1b[0m")
            print ("Parameters are :")
            print ("\t \tlr = %f\n\
                    beta1 = %f\n\
                    beta2 = %f\n\
                    decay = %f" % (par.lr, par.beta1, par.beta2, par.decay)
                  )
                  
            if bool(input("Do you want to change them (1/0)? " )) :
                kwargs["lr"] = float(input("lr (default = %f): " % (default_value[par.opti]["lr"])))
                kwargs["beta1"] = float(input("beta1 (default = %f): " % (default_value[par.opti]["beta1"])))
                kwargs["beta2"] = float(input("beta2 (default = %f): " % (default_value[par.opti]["beta2"])))
                kwargs["decay"] = float(input("decay (default = %f): " % (default_value[par.opti]["decay"])))

                print ("\x1b[1;37;43m%s: lr = %f, beta1 = %f, beta2 = %f, decay = %f\x1b[0m"\
                            % (par.opti, kwargs["lr"], kwargs["beta1"],\
                               kwargs["beta2"], kwargs["decays"], )  
                      )
            else :
                kwargs["lr"] = par.lr
                kwargs["beta1"] = par.beta1
                kwargs["beta2"] = par.beta2
                kwargs["decay"] = par.decay
        
        if par.opti == "Nadam" :
            print ("\x1b[1;37;44mNadam Choosen\x1b[0m")
            print ("Parameters are :")
            print ("\t \tlr = %f\n\
                    beta1 = %f\n\
                    beta2 = %f\n\
                    schedule_decay = %f" % (2*par.lr, par.beta1, par.beta2, par.schedule_decay)
                  )
                  
            if bool(input("Do you want to change them (1/0)? " )) :
                kwargs["lr"] = float(input("lr (default = %f): " % (default_value[par.opti]["lr"])))
                kwargs["beta1"] = float(input("beta1 (default = %f): " % (default_value[par.opti]["beta1"])))
                kwargs["beta2"] = float(input("beta2 (default = %f): " % (default_value[par.opti]["beta2"])))
                kwargs["schedule_decay"] = float(input("schedule_decay (default = %f): " % (default_value[par.opti]["schedule_decay"])))

                print ("\x1b[1;37;43m%s: lr = %f, beta1 = %f, beta2 = %f, schedule_decay = %f\x1b[0m"\
                            % (par.opti, kwargs["lr"], kwargs["beta1"],\
                               kwargs["beta2"], kwargs["schedule_decay"])  
                      )
            else :
                kwargs["lr"] = 2*par.lr
                kwargs["beta1"] = par.beta1
                kwargs["beta2"] = par.beta2
                kwargs["schedule_decay"] = par.schedule_decay
        return kwargs 
    
    kwargs = defined_optimizer(parser)
    print ("\n\x1b[1;19;10mkwargs to lunch the NN :\n{}\x1b[0m".format(kwargs)) 
    print ("-"*20)
    
#    kwargs = 
    from sklearn.datasets import load_boston
    X, y = load_boston().data, load_boston().target
    
    datapath = "./../cases/data/burger_dataset/burger_matrices"    
    
    
    dict_layers = {"I" : 4,\
                   "N1" : [100,"relu"],\
                   "N2" : [100,"relu"],\
                   "N3" : [100,"relu"],\
                   "N4" : [100,"relu"],\
                   "N5" : [100,"relu"],\
                   "N6" : [100,"relu"],\
                   "N7" : [100,"relu"],\
                   "N8" : [100,"relu"],\
                   "N9" : [100,"relu"],\
                   "N10" : [100,"relu"],\
                   "O"  : [1, "relu"]\
                  }
    
    kwargs["batch_size"] = par.bsz
    
    k_nn = K_Neural_Network(dict_layers, opti=par.opti, loss=par.loss, metrics=par.metrics, max_epoch=par.maxepoch,\
                            verbose=False, non_uniform_act=True, **kwargs) 

    k_nn.train_and_split(X, y, shuffle=True, scale=True)
    k_nn.build()
    
    k_nn.compile(save=True)
    
#    k_nn.model.save("keras_model.h5")
    
#    run Class_Keras_NN.py -lr 5e-4 -opti Adam -beta1 0.8
    
#    import matplotlib as mtp
#    plt.figure()
#    plt.scatter(k_nn.y_test, k_nn.y_test, marker = mtp.markers.MarkerStyle(marker='o', fillstyle='none'), c='black')
#    plt.scatter(k_nn.y_test, k_nn.prediction(k_nn.X_test), c='green', marker='+')

