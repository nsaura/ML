#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.ion()

s = pd.Series([909976, 8615246, 2872086, 2273305])

print "s.index: {} \ns.values: {}".format(s.index, s.values)

s.index= ["Stockholm", "London", "Rome", "Paris"]
s.name = "Population"

#In [10]: s
#Out[10]: 
#Stockholm     909976
#London       8615246
#Rome         2872086
#Paris        2273305
#Name: Population, dtype: int64

## On peut aussi faire :
s = pd.Series([909976, 8615246, 2872086, 2273305], name="Population", index=["Stockholm", "London", "Rome", "Paris"])

print("s.median = {} \t s.mean = {} \t s.std = {}".format(s.median(), s.mean(), s.std()))
print("s.min = {} \t s.max = {}".format(s.min(), s.max()))

print("s.quantile(0.25) = {} \t s.quantile(0.5) = {} \t s.quantile(0.75) = {}".format(s.quantile(0.25), s.quantile(0.5), s.quantile(0.75)))

## Tout cela se retrouve dans 
print("s.describe: \n{}".format(s.describe()))

verbose = False
if verbose == True :
    fig, axes = plt.subplots(1, 4, figsize=(17, 6))
    s.plot(ax=axes[0], kind='line', title='line')
    s.plot(ax=axes[1], kind='bar', title='bar')
    s.plot(ax=axes[2], kind='box', title='box')
    s.plot(ax=axes[3], kind='pie', title='pie') # Camembert

## Finally series is a relevant container for one dimensional array
# For higher dimensional arrays we use the dataFrame structure :

## Plusieurs façons d'initialiser :
df = pd.DataFrame(  [[909976,     "Sweden"],
                     [8615246,    "United-Kingdom"],
                     [2872086,    "Italy"  ],
                     [2273305,    "France"]] )

#               Columns                     
# index         0          1
#           0   909976  Stockholm
#           1  8615246     London
#           2  2872086       Rome
#           3  2273305      Paris

df.index    =   ["Stockholm", "London", "Rome", "Paris"]
df.columns  =   ["Population", "State"]

# Ou encore

df = pd.DataFrame(  [[909976,     "Sweden"],
                     [8615246,    "United-Kingdom"],
                     [2872086,    "Italy"  ],
                     [2273305,    "France"]],
                    columns =   ["Populations", "State"],
                    index   =   ["Stockholm", "London", "Rome", "Paris"]    )

#df = pd.DataFrame(  {"Population": [909976, 8615246, 2872086, 2273305],
#                            "State": ["Sweden", "United Kingdom", "Italy", "France"]},
#                     index=["Stockholm", "London", "Rome", "Paris"] )

## On peut utiliser l'indexer df.ix :
#  the ix indexer results in a new DataFrame that is a subset of the original
print("df.ix[\"Stockholm\"] :\n{}".format(df.ix["Stockholm"]))
print("df.ix[[\"Paris\", \"Rome\"]] :\n{}".format(df.ix[["Paris", "Rome"]]))

print(df.info())

df_pop = pd.read_csv("european_cities.csv", delimiter=",", encoding="utf-8", header=0)

# lambda x : int(x.replace(",", "")) va supprimer les , dans population et remplacer par rien
# puis change le type du résultat en int (initialement str)
df_pop["NumericPopulation"] = df_pop.Population.apply(lambda x : int(x.replace(",", "")))
df_pop["State"] = df_pop["State"].apply(lambda x: x.strip())

city_counts = df_pop.State.value_counts()

df_pop2 = df_pop.set_index("City") # remplace les 0 1 2 etc qui étaient en index.
# Exemple :

#In [52]: df_pop
#Out[52]: 
#    Rank         City           State Population     Date of census  \
#0      1       London  United-Kingdom  8,787,892       22 June 2016   
#1      2       Berlin         Germany  3,670,622   31 december 2016   
#In [53]: df_pop2
#Out[53]: 
#             Rank           State Population     Date of census  \
#City                                                              
# London         1  United-Kingdom  8,787,892       22 June 2016   
# Berlin         2         Germany  3,670,622   31 december 2016   

# Trie par ordre alphabetique de l'état (hiérarchisation State, city et autre)
df_pop3 = df_pop.set_index(["State", "City"]).sortlevel(0) 

#In [56]: df_pop3
#Out[56]: 
#                            Rank Population     Date of census  \
#State          City                                              
#Austria         Vienna         7  1,877,836        1 July 2017   
#Belgium         Brussels      16  1,187,890     1 January 2016   

#In [58]: df_pop3.ix["Spain"]
#Out[58]: 
#            Rank Population    Date of census  NumericPopulation
#City                                                            
# Barcelona    11  1,608,746  31 december 2016            1608746
# Madrid        3  3,141,991    1 January 2015            3141991

#In [79]:  df_pop3.ix[("France", "Paris")]
#Out[79]: 
#Rank                              5
#Population                2,244,000
#Date of census       1 January 2015
#NumericPopulation           2244000
#Name: (France, Paris), dtype: object

# Hierarchisation State, City, NumericPopulation
# Dans l'ordre des index
df_pop3 = df_pop[["State", "City", "NumericPopulation"]].set_index(["State", "City"])

#Somme de la populations des villes appartenant à chaque pays puis les classe de manière décroissante
df_pop4 = df_pop3.sum(level="State").sort("NumericPopulation", ascending=False) 
#!/usr/bin/env python
#-*- coding:utf-8 -*-

# .drop("Name", axis = 0 ou 1) drop permet d'enlever la ligne ou la colonne (axis =0 ou 1) s'appelant Name
# groupby : creation d'une nouvelle DataFrame regroupée avec la clé dans dans groupby
# .sum() : fait la somme dans la dataframe 
df_pop5 = (df_pop.drop("Rank", axis=1).groupby("State").sum().sort("NumericPopulation", ascending=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
city_counts.plot(kind='barh', ax=ax1)
ax1.set_xlabel("# cities in top 105")
df_pop5.NumericPopulation.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Total pop. in top 105 cities")

