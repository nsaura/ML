#!/usr/bin/python
# -*- coding: latin-1 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.ion()

pd.date_range("2015-1-1", periods=31)
#DatetimeIndex(['2015-01-01', '2015-01-02', '2015-01-03',. .. '2015-01-31',
#              dtype='datetime64[ns]', freq='D', tz=None)

pd.date_range("2015-1-1 00:00", "2015-1-1 12:00", freq="H")
#DatetimeIndex(['2015-01-01 00:00:00', '2015-01-01 01:00:00',
#               '2015-01-01 02:00:00', ...'2015-01-01 12:00:00'],
#              dtype='datetime64[ns]', freq='H', tz=None)

ts1 = pd.Series(np.arange(31), index=pd.date_range("2015-1-1", periods=31))
#2015-01-01     0
#2015-01-02     1
#    .
#    .
#    .
#2015-01-30    29
#2015-01-31    30
#Freq: D, dtype: int64 
#Freq: D pour DAY

(year, month, day) = (ts1.index[2].year, ts1.index[2].month, ts1.index[2].day)

ts1.index[2].to_pydatetime()
#datetime.datetime? datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
print ts1.index[2].to_pydatetime()

import datetime
ts2 = pd.Series(np.random.rand(2), index=[datetime.datetime(2015,1,1), datetime.datetime(2015,1,1)])
#2015-01-01    0.810779
#2015-01-01    0.652360
#dtype: float64

#We can use datetime.datetime ou bien pd.PeriodIndex :
periods = pd.PeriodIndex([pd.Period('2015-01'),
                          pd.Period('2015-02'),
                          pd.Period('2015-03')])
ts3 = pd.Series(np.random.rand(3), index=periods) 
#2015-01    0.535307
#2015-02    0.788048
#2015-03    0.613808

# On peut même récupérer la fréquence etc.
print ts3.index
#PeriodIndex(['2015-01', '2015-02', '2015-03'], dtype='int64', freq='M')

#On peut convertir une serie en PeriodIndex 
ts2.to_period('M')

 def plot_rv_distribution(X, axes=None):
 """Plot the PDF or PMF, CDF, SF and PPF of a given random variable"""
 if axes is None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    x_min_999, x_max_999 = X.interval(0.999)
    x999 = np.linspace(x_min_999, x_max_999, 1000)
    x_min_95, x_max_95 = X.interval(0.95)
    x95 = np.linspace(x_min_95, x_max_95, 1000)
    if hasattr(X.dist, "pdf"):
    axes[0].plot(x999, X.pdf(x999), label="PDF")
    axes[0].fill_between(x95, X.pdf(x95), alpha=0.25)
    else:
    # discrete random variables do not have a pdf method, instead we use pmf:
    x999_int = np.unique(x999.astype(int))
    axes[0].bar(x999_int, X.pmf(x999_int), label="PMF")
    axes[1].plot(x999, X.cdf(x999), label="CDF")
    axes[1].plot(x999, X.sf(x999), label="SF")
    axes[2].plot(x999, X.ppf(x999), label="PPF")
    for ax in axes:
        ax.legend()

             
