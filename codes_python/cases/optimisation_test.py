#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv, os, sys

plt.ion()

T_recup = []
T_inf_lst = [i*5 for i in xrange(1, 11)]

N_discr = 33

line_z = np.linspace(0.,1.,N_discr)[1:N_discr-1]

for i, T_inf in enumerate(T_inf_lst) : 
    #i : xrange(len(T_inf_lst), T_inf = T_inf_lst[i]
    pathfile = os.path.join('./data', 'T_inf_{}.csv'.format(T_inf))
    T_temp = pd.read_csv(pathfile).get_values()
    T_recup.append(T_temp.reshape(T_temp.shape[0]))

for i, (T_inf, TT) in enumerate(zip(T_inf_lst,T_recup)) :
    if i%3 == 0 :
        plt.plot(line_z, TT, label="T_inf: {}".format(T_inf))

plt.legend(loc='best')


