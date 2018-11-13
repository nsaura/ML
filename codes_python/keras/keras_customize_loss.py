#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

#https://stackoverflow.com/questions/46858016/keras-custom-loss-function-to-pass-arguments-other-than-y-true-and-y-pred?noredirect=1&lq=1

import numpy as np
from keras import backend as K
from keras.metrics import mse

def Ridgeloss(weights, val = 0.01):
    def lossFunction(y_true, y_pred) :
        loss = mse(y_true, y_pred)
        loss += K.prod(val, K.sum(K.square(weights)))

def LassoLoss(weights, val=1e-7) :
    def lossFunction(y_true, y_pred) :
        loss = mse(y_true, y_pred)
        loss += K.prod(val, K.sum(K.abs(weights)))

# To use one of them
# model.compile(loss=customLoss(weights,0.03), optimizer =..., metrics = ...) 

