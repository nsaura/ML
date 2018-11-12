#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
from csv import reader
from matplotlib.pyplot import plot, legend, figure, ion
from os.path import splitext, split, abspath

ion()
def plot_from_file(path, delimiter='\n', rescale=1.) :
    """
    plot value in a file
    """
    path = abspath(path)
    file = reader(open(path, "r"), delimiter=delimiter)
    values = []
    for l in file :
        values.append(float(l[0])/rescale)
    
    figure("Evolution de %s" %(split(splitext(path)[0])[1]))
    plot(range(len(values)), values, label="Evolution de %s" %(split(splitext(path)[0])[1]))
    legend()
