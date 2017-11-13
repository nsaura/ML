#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def tab_normal(mu, sigma, length) :
    return s

def test (**kwargs) :
    dico = dict()
    for kway in kwargs :
        dico[kway] = kwargs[kway]
    
    print dico['thrd']

N_discr=50; T_inf=50 
line_z = np.linspace(0.,1.,N_discr)
T_n =  list(map(lambda x : -4*T_inf*x*(x-1), line_z[1:N_discr-1]))

df = pd.DataFrame(T_n)    
df.to_csv("T.csv", index=False, header=True)

T_read = pd.read_csv("T.csv").get_values()
T_read = T_read.reshape(T_read.shape[0])

lii = line_z[1:N_discr-1]

plt.ion()
plt.figure("writen and read")
plt.plot(lii, T_read, label='read', marker='o')
plt.plot(lii, T_n, label='written')
plt.legend(loc='best')
plt.show() 
    



