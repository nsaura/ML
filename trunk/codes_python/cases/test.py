#!/usr/bin/python
# -*- coding: latin-1 -*-

from scipy.stats import norm as norm 
import class_temp_ML as ctml
ctml = reload(ctml)


p = ctml.parser()
T = ctml.Temperature(p)

x = T.tab_normal(0,0.1,1000)
T.get_prior_statistics()
#norm.pdf(x)



