#!/usr/bin/python
# -*- coding: latin-1 -*-
from __future__ import division

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns

import os
import os.path as osp

import numdifftools as nd

import tensorflow as tf

from threading import Thread, RLock

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

import time


verrou= RLock()

dx = 0.037037037037037035
dt = 0.014814814814814815
nu = 0.025
CFL = 0.4
Nx = 81
L = 3
line_x = np.arange(0,L,dx)

#dict_layers = {"inputs" : [1], "N1" : [20, "tanh"], "N2" : ["20", "tanh"], "action":[1]}

def LW_solver(u_init, Nx, dx, dt, nu) :
    def intermediaires (var, flux ,incr, r) :
        """ Fonction qui calcule les valeurs de la variables  aux points (n + 1/2 , i - 1/2) et (n + 1/2 , i + 1/2)   
            Parametres :
            -------------
            var : Celle dont on veut calculer l'étape intermediaire. C'est un tableau de valeur
            flux : le flux correspondant à la valeur var        
            incr : l'indice en cours indispensable pour evaluer l'etape intermediaire
            r : rapport dt/dx
            
            Retour : 
            ------------
            f_m = la valeur intermediaire au point (n + 1/2 , i - 1/2)
            f_p = la valeur intermediaire au point (n + 1/2 , i + 1/2)
        """
        f_m = 0.5 * ( var[incr] + var[incr-1] ) - 0.5 * r * ( flux[incr]- flux[incr-1] )
        f_p = 0.5 * ( var[incr+1] + var[incr] ) - 0.5 * r * ( flux[incr+1] - flux[incr] )
        return f_m,f_p
    
    r = dt / dx
    t = it = 0
    u = np.copy(u_init)
    u_nNext = []
    
    fac = nu * dt / dx **2
    
    fu = np.asarray([0.5*u_x**2 for u_x in u])
    
    der_sec = [fac*(u[k+1] - 2*u[k] + u[k-1]) for k in range(1, len(u)-1)]
    der_sec.insert(0, fac*(u[1] - 2*u[0] + u[-1]))
    der_sec.insert(len(der_sec), fac*(u[0] - 2*u[-1] + u[-2]))
    
    for i in range(1, Nx-1) :
        u_m, u_p = intermediaires(u, fu, i, r)
        fu_m =  0.5*u_m**2
        fu_p =  0.5*u_p**2

        u_nNext.append( u[i] - r*( fu_p - fu_m ) + der_sec[i] )
                                    
    # Conditions aux limites 
    u[1:Nx-1] = u_nNext  
    u_nNext  = []
    
    u[0] = u[-2]
    u[-1]= u[1]
    
    u = np.asarray(u) 
        
    return u

###-------------------------------------------------------------------------------

#class Calcul_gradient(Thread) :
#    def __init__(self, grad):
#        Thread.__init__(self)
#        self.grad = grad
#    
#    def set_field(self, field) :
#        self.field = field
#    
#    def run(self):
#        return self.grad(self.field)
###-------------------------------------------------------------------------------                
###-------------------------------------------------------------------------------                
    
class actor_deepNN() :
    def __init__(self, dict_layers, lr, N_thread = 10, op_name="Adam", loss_name="OLS", reduce_type="sum", case_filename='./../cases/multinn_forwardNN/data/burger_dataset/'):
        nodes, acts = {}, {}
        
        act_func_lst = ["relu", "sigmoid", "tanh", "leakyrelu", "selu", "swish"]
        
        act_str_to_tf = {"elu"      :   tf.nn.elu,
                         "relu"     :   tf.nn.relu,
                         "tanh"     :   tf.nn.tanh,
                         "selu"     :   tf.nn.selu,
                         "sigmoid"  :   tf.nn.sigmoid,
                         "leakyrelu":   tf.nn.leaky_relu
                        }
        
        opt_str_to_tf = {"Adam"     :   tf.train.AdamOptimizer,
                         "RMS"      :   tf.train.RMSPropOptimizer,
                         "Momentum" :   tf.train.MomentumOptimizer,
                         "GD"       :   tf.train.GradientDescentOptimizer
                        }
        
        for item in dict_layers.iteritems() :
            if item[0] == "inputs" or item[0] == "action" :
                nodes[item[0]] = int(item[1][0])
            else :
                nodes["A_hn"+item[0][1:]] = int(item[1][0])
                acts["A_act"+item[0][1:]] = act_str_to_tf[str(item[1][1])]
        
        self.acts = acts
        self.nodes = nodes
        
        s_dim = dict_layers["inputs"][0]
        
        self.dict_layers = dict_layers
        self.N_thread = N_thread

        self.state = tf.placeholder(tf.float32, shape=(None, s_dim), name='inputs')
        self.action = tf.placeholder(tf.float32, shape=(None, s_dim), name='action')


        self.sess = tf.InteractiveSession(config=config)

        self.act_func_lst = act_func_lst
        
        self.last_key = lambda string: "A_%s_Last" % (string)
        self.last_name = lambda string: "Action_%s_Last" % (string)
        
        self.layer_key = lambda string, number : "A_%s%d" % (string, number)
        self.layer_name = lambda string, number : "Action_%s_%d" % (string, number)
        
        self.lr = lr
        self.op_name = op_name
        self.opt_str_to_tf = opt_str_to_tf
        
        self.loss_name = loss_name
        self.case_filename = case_filename
        
        self.reduce_fct = tf.reduce_sum if reduce_type == "sum" else tf.reduce_mean
        
        self.init_NN_Actor_parameter()
        self.build_NN_operating_Actor_graph()
        
        self.def_optimizer()
        
        self.network_params = tf.trainable_variables()
        
        # Op for periodically updating target network with online network
        # weights
#        self.update_target_network_params = \
#            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
#                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
#            for i in range(len(self.target_network_params))]
        
        self.action_gradient = tf.placeholder(tf.float32, (None, s_dim))
        
#        https://datascience.stackexchange.com/questions/18695/understanding-the-training-phase-of-the-tutorial-using-keras-and-deep-determini/18705#18705
        self.actor_gradients = tf.gradients(self.NN_guess, self.network_params, -self.action_gradient) # On multiplie le dernier grdient par le gradient de l'action
        
        self.optimize = self.tf_optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
###-------------------------------------------------------------------------------

    def init_NN_Actor_parameter(self) :
        
        prev_key = "inputs"
        weights = dict()
        
        for nlayer in range(1, len(self.acts.keys())+1) :
            prev_node = self.nodes[prev_key]
            curr_node_key = self.layer_key("hn", nlayer)

            curr_node = self.nodes[curr_node_key]
            
#            print ("prev_key : {}, type = {}".format(prev_key, type(prev_key)))
#            print ("prev_node : {}, type = {}".format(prev_node, type(prev_node)))
#            
#            print ("curr_node : {}, type = {}".format(curr_node, type(curr_node)))
#            print ("\n")
            
            tempWeight = np.random.randn(prev_node, curr_node) / prev_node
            
            dict_key = self.layer_key("w", nlayer)
            weights_tf_name = self.layer_name("Weight", nlayer)
            
            weights[dict_key] = tf.Variable(tempWeight.astype(np.float32), name=weights_tf_name)
            
            prev_key = curr_node_key
        
        prev_node = self.nodes[prev_key]
        curr_node = self.nodes["action"]
        
        tempWeight = np.random.randn(prev_node, curr_node) / prev_node
        
        weights[self.last_key("w")] = tf.Variable(tempWeight.astype(np.float32),\
                                         name=self.last_name("Weight"))
        
        #La boucle est bouclée
        
        for item in weights.iteritems() :
            print ("item : {}, shape = {}".format(item[0], np.shape(item[1])))
        
        self.weights = weights
        
###-------------------------------------------------------------------------------
    
    def build_NN_operating_Actor_graph(self, tau=0.0001) :
        action_layers = dict()
        incoming_layer = self.state
        
        for nlayer in range(1, len(self.acts.keys())+1) :
            w_key = self.layer_key("w", nlayer)
            a_key = self.layer_key("act", nlayer)
            
            action_layer = tf.matmul(incoming_layer, self.weights[w_key])
            action_layer = self.acts[a_key](action_layer, name=self.layer_name("act", nlayer))
            
            action_layers[a_key] = action_layer
            
            incoming_layer = action_layer
        
        last_action_layer = tf.matmul(incoming_layer, self.weights[self.last_key("w")])
        action_layers[self.last_key("act")] = self.acts[a_key](last_action_layer,\
                                            self.last_name("act"))
        
        print action_layers

        self.NN_guess = last_action_layer
        self.action_layers = action_layers
        
###-------------------------------------------------------------------------------

    def def_optimizer(self, decay=0.9, beta1=0.9, beta2=0.99, momentum=0.0, nest=False):
        
        tf_optimizer = self.opt_str_to_tf[self.op_name]
        
        if self.op_name == "GD" :
            tf_optimizer = tf_optimizer(self.lr, name=self.op_name)
        
        if self.op_name == "Momentum" :
            tf_optimizer = tf_optimizer(self.lr, momentum, use_nesterov=nest, name=self.op_name)
                    
        if self.op_name == "Adam" :
            tf_optimizer = tf_optimizer(self.lr, beta1, beta2, name=self.op_name)
        
        if self.op_name == "RMS" : 
            tf_optimizer = tf_optimizer(self.lr, decay, momentum, name=self.op_name)
        
        self.tf_optimizer = tf_optimizer
        
###-------------------------------------------------------------------------------                
     
    def update_target_network(self, tau=0.001) : 
        self.sess.run(self.update_target_network_params())        
       
###-------------------------------------------------------------------------------        
    
    def A_prediction(self, state) :
        return self.sess.run(self.NN_guess, feed_dict={self.state : state}).reshape(-1)

###-------------------------------------------------------------------------------

    def rewards (self, action):
        penalty = np.zeros((action.ravel().size))
        
        for j in range(1, len(action)-1) :
            temp = (action[j] - self.prev[j])/self.dt
            
            NL_t = self.prev[j] * (self.prev[j+1] - self.prev[j-1])/(2*self.dx)
            
            diff = self.nu*(self.prev[j+1] - 2*self.prev[j] + self.prev[j-1])/self.dx**2
            
            penalty[j] = temp + NL_t - diff
        
        penalty[0] = penalty[-2]
        penalty[-1] = penalty[1]
    
        return np.mean(-penalty)
    
###-------------------------------------------------------------------------------    
    def DRL(self, itmax, Nx, dx, dt, nu, line_x, L):
        
        true_action = np.zeros((0))
        self.dt = dt
        self.dx = dx
        self.nu = nu
        
        rewards = lambda action : -(true_action - action).T.dot(true_action - action)
        dJ = nd.Gradient(rewards)      
                
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        true_incoming_field = np.sin(2*np.pi*line_x/L)
        pred_incoming_field = np.sin(2*np.pi*line_x/L)
        
        fig, axes = plt.subplots(1, 2, figsize=(8,8))
        
        for it in range(itmax) :
            ranged_2 = 50
            ranged = 5
            for k in range(ranged_2) : 
                colorss=iter(cm.spectral_r(np.arange(ranged_2)))

                self.prev = pred_incoming_field
                it_pred = self.A_prediction(state=pred_incoming_field.reshape(1,-1))
        
                true_action = LW_solver(true_incoming_field, Nx, dx, dt, nu)
            
                print ("it_pred it = {}:\n{}".format(it, it_pred))
                if k ==0 :
                    prev_rew_lst = [0 for r in range(ranged)]
                else :
                    prev_rew_lst = [r*0.95 for r in rew_lst]
                rew_lst = []
                gra_lst = []
                gra_thread_lst = []
            
            
#            n_rotation = ranged // self.N_thread
#            leftover = ranged % self.N_thread
#            
#            print ("N_rotation = {}\nLeftover={}".format(n_rotation, leftover))
#            
#            thread_dict = {}
#            for thread in range(self.N_thread) :
#                thread_dict["thread_%d" % thread] = Calcul_gradient(dJ)
#            
#            self.thread_dict = thread_dict
#            
#            time_1 = time.time()
#            
#            for rep in range(n_rotation) :
#                for titem in thread_dict.iteritems() :  
#                    it_pred = self.A_prediction(state=(pred_incoming_field.reshape(1,-1) + np.random.normal(0,1,Nx)))    
##                    titem[1].set_field(it_pred)
#                    gra_thread_lst.append(titem[1].run(it_pred))
#                    
##                    print titem[0] 
#                    
#                print np.shape(gra_thread_lst)
##                
##                for thread in range(self.N_thread) :
##                    print ("thread_{} : {}".format(thread, thread_dict["thread_%d" % thread]))
##                    thread_dict["thread_%d" % thread].join()
#            
#            for rep in range(leftover) :
#                it_pred = self.A_prediction(state=(pred_incoming_field.reshape(1,-1) + np.random.normal(0,1,Nx)))    
##                thread_dict["thread_%d" % thread].set_field(it_pred)
#                gra_thread_lst.append(thread_dict["thread_%d" % thread].run(it_pred))
#                
#            
##            for thread in range(leftover) :
##                try :
##                    thread_dict["thread_%d" % thread].join()
##                except :
##                    pass

                time_2 = time.time()
#            print ("Time for multithreading part = {}s".format(abs(time_1-time_2)))
            
                for rep in range(ranged) :
                    it_pred = self.A_prediction(state=(pred_incoming_field.reshape(1,-1) + np.random.normal(0,1,Nx)))    
                    rew_lst.append(rewards(it_pred)+prev_rew_lst[rep])
                    gra_lst.append(dJ(it_pred))
                    
                    # Pour pouvoir évaluer une moyenne du gradient
                
                time_3 = time.time()            
                print ("temps de calcul = {}s".format(abs(time_3 - time_2)))
                print rew_lst
                
                grads_to_apply = np.mean([rew_lst[i]*gra_lst[i] for i in range(ranged)])
                vectorized_grads = np.full((1, np.shape(true_action)[0]), grads_to_apply)
                
                print ("grad = {}:\n".format(grads_to_apply))
                
                c= next(colorss)
                axes[0].plot(it, grads_to_apply, color=c, marker="o", linestyle='none')
                axes[0].set_title("it %d, it %d" % (it, k))
                
                plt.pause(0.01)
                
                self.sess.run(self.optimize, feed_dict={self.state : pred_incoming_field.reshape(1,-1),
                                                        self.action_gradient : vectorized_grads})
                
    #            it_pred_2 = self.A_prediction(state=pred_incoming_field.reshape(1,-1))
    #            reward_2 = self.rewards(it_pred_2)
    #            
    #            axes[0].semilogy(it, reward_2, color="navy", marker="o", linestyle='none')
    #            
    #            axes[0].legend(["Red : First estimation", "Blue : Second estimation"])
    #            
#                axes[1].cla()
                if k == 0 :
                    axes[1].plot(line_x, true_action, label="True value it %d" %it, color="blue")
                axes[1].plot(line_x, it_pred, label="First value", linestyle="none", marker="o", c=c, fillstyle="none")
    #            axes[1].plot(line_x, it_pred_2, label="second Pred value", linestyle="none", marker="o", c='crimson', fillstyle="none")
    #            
    #            axes[1].legend()            
    #            
                plt.pause(1)
                
                [self.network_params[i].assign(tf.multiply(self.network_params[i], 0.95)) for i in range(len(self.network_params))]
                        
            true_incoming_field = true_action
            pred_incoming_field = it_pred
            
#            print ("weights = \n{}".format(self.sess.run(tf.trainable_variables()[0])))
            
            time.sleep(5)
###------------------------------------------------------------------------------- 

if __name__=="__main__" :
#    run ReiLFramework_Actor_block.py
    try :
        tf.reset_default_graph()
    except :
        print ("This time the graph won't be deleted")
        pass
    
    dict_layers = {"inputs" : [81], "N1" : [5, "sigmoid"],# "N2" : [20, "sigmoid"],  "N3" : [20, "sigmoid"], "N4" : [20, "sigmoid"],
                   "action":[81]}
    
    actor = actor_deepNN(dict_layers, 1e-2)
    actor.init_NN_Actor_parameter()
    actor.build_NN_operating_Actor_graph()
    actor.def_optimizer(beta2=0.5)

####-------------------------------------------------------------------------------

#    def def_loss(self, r=1, alpha=1e-7) :
#        
#        classic_loss = {"OLS" = tf.square, "AVL" : tf.abs}
#        regress_loss = ["Lasso", "Ridge", "Elastic"]
#        
#        if self.loss_name in classic_loss.keys() :
#            loss_function = classic_loss[self.loss_name]
#        
#        else :
#            if self.loss_name == "Lasso" :
#                loss_function = self.Elastic_cost(r=1, alpha=alpha)
#            
#            if self.loss_name == "Ridge" :
#                loss_function = self.Elastic_cost(r=0, alpha=alpha)
#            
#            if self.loss_name == "Elastic" :
#                loss_function = self.Elastic_cost(r=r, alpha=alpha)
#        
#        self.loss_name = self.reduce_fct(loss_function)
#        
####-------------------------------------------------------------------------------        
#        
#    def Elastic_cost(self, r=1) :
#        # Pseudo code : 
#        # J = MSE + r*alpha * sum(|weights|) + (1-r)*0.5*alpha * sum(tf.square(weights))
#        # if r == 0 ---> Ridge
#        # if r == 1 ---> Lasso 
#        
#        if np.abs(r) < 1e-8 :
#            print ("r = %f --> Ridge function selected" % r)
#        
#        if np.abs(r-1) < 1e-8 :
#            print ("r = %f --> Lasso function selected" % r)
#        
#        self.weight_sum = tf.placeholder(np.float32, (None), name="weight_sum")
#        
#        MSE_part = self.reduce_type_fct(tf.square(self.NN_guess - self.action))
#        
#        Ela_part = tf.add(tf.multiply(r*alpha, self.reduce_type_fct(tf.abs(self.weight_sum))),
#               tf.multiply((1.-r)*0.5*alpha, self.reduce_type_fct(tf.square(self.weight_sum))))
#        
#        return tf.add(MSE_part, Ela_part) 

        
#        log_path = os.path.abspath("./logs")
#        now = datetime.utcnow().strftime("%Y_%m_%d_%Hh%m_%S")

#        dir_name = os.path.join(log_path, now)
#        
#        log_dir = "{}/run-{}".format(dir_name, now)
#        
#        self.log_dir = log_dir
#        self.dir_name = dir_name
#                    
#        file_writer = tflearn.summaries.get_summary(self.log_dir, tf.get_default_graph()) 
#        file_writer.close()   
                        
        
