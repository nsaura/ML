# load and plot dataset
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt


plt.ion()
# load dataset
series = read_csv('./Datasets/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

# summarize first few rows
print(series.head())
# line plot
series.plot()
plt.show()

# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]

#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
	
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
