#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, warnings, argparse

import os
import os.path as osp

from scipy import optimize as op
from itertools import cycle
import matplotlib.cm as cm

import numdifftools as nd

import time
import glob

## ---- Description et aide sur la méthode 

# Nom sous la forme str:value_str:value" etc
# valeur en float sous la forme de e-1 par exemple
# Puis on split par rapport à _ et on construit une liste de split par rapport à : exemple 

#q = "g:12_h:45".split("_")
#l = [i.split(":") for i in q]

#print ("str = g:12_h:45")

#print ("\nsplit(\"_\")")
#print (q)

#print ("\nsplit(\":\")")
#print (l)

##                                      -----------------
import Class_Vit_Choc as cvc

parser = cvc.parser()
cb = cvc.Vitesse_Choc(parser)

u_name = cb.u_name
b_name = cb.beta_name


betaloc = "./data/burger_dataset/betas/*"
files = glob.glob(betaloc)

print osp.split(files[0])[1].split("_")

l = osp.split(files[0])[1].split("_")

ll = [i.split(":") for i in l[1:-1]]

dico = dict()
for elt in ll :
    dico[elt[0]] = []

for f in files :
    l = osp.split(f)[1].split("_") 
    ll = [i.split(":") for i in l[1:-1]]
    for elt in ll :
        if elt[1] not in dico[elt[0]] :
            dico[elt[0]].append(elt[1])
print dico

lst_pairs = []
for f in files :
    if "U" in f :
        f_to_find = osp.split(f)[0] + "/beta" + osp.split(f)[1][1:]
        if osp.exists(f_to_find) == True :
            lst_pairs.append((files[files.index(f_to_find)], f))
        else :
            print ("%s does not exist" %(f_to_find))

for i in range (10) :
    print lst_pairs[np.random.randint(len(lst_pairs))]

X = np.zeros((3))
y = np.zeros((1))

# pairs :   p[0] -> beta
#           p[1] -> u
for p in lst_pairs :
    u = cb.pd_read_csv(p[1])
    beta = cb.pd_read_csv(p[0])
    for j in range(1, len(u)-1) :
        X = np.block([[X], [u[j-1], u[j], u[j+1]]])
        y = np.block([[y], [beta[j]]])
        
X = np.delete(X, 0, axis=0)
y = np.delete(y, 0, axis=0)
