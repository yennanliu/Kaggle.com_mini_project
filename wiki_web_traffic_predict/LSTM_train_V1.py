# ops 
import numpy as np
import pandas as pd 
import datetime as dt
import time
import math 
import re

# DL 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback



def load_data():
	train = pd.read_csv('train_1.csv').fillna(0)
	return train 


# help functions 

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        """
        ----- fix here for python 3 ----
        https://stackoverflow.com/questions/18493677/how-do-i-return-a-string-from-a-regex-match-in-python
        """
        return res.group(0)[:2]
    return 'na'


def get_aggregated_data(train):
	lang_sets = {}     # get the search data without language column 
	lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
	lang_sets['ja'] = train[train.lang=='ja'].iloc[:,0:-1]
	lang_sets['de'] = train[train.lang=='de'].iloc[:,0:-1]
	lang_sets['na'] = train[train.lang=='na'].iloc[:,0:-1]
	lang_sets['fr'] = train[train.lang=='fr'].iloc[:,0:-1]
	lang_sets['zh'] = train[train.lang=='zh'].iloc[:,0:-1]
	lang_sets['ru'] = train[train.lang=='ru'].iloc[:,0:-1]
	lang_sets['es'] = train[train.lang=='es'].iloc[:,0:-1]

	sums = {}         # avg daily searching (for each language )
	for key in lang_sets:
	    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
	print (sums)
	return sums 

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



def single_input_LSTM(sums,language):
	#for language in sums.keys():
	#for language in ['fr']:
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(sums[language].reshape(-1, 1))
	print ('language : ', language)
	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	print ('-------')
	print ('train_size : ', train_size)
	print ('test_size : ', test_size)
	print ('-------')
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	#plot
	"""
	series,=plt.plot(scaler.inverse_transform(dataset)[:,])  
	prediccion_entrenamiento,=plt.plot(trainPredictPlot[:,],color = 'red')  
	prediccion_test,=plt.plot(testPredictPlot[:,],color = 'blue')
	plt.title('Web View Forecasting (LSTM, lookback=1)')
	plt.xlabel('Number of Days from Start')
	plt.ylabel('Web View')
	plt.legend()
	plt.legend([serie,prediccion_entrenamiento,prediccion_test],['all data','train','test'], loc='upper right')
	plt.show()
	"""




if __name__ == '__main__':
	train = load_data()
	train['lang'] = train.Page.map(get_language)
	print (train.head(3))
	sums = get_aggregated_data(train)
	single_input_LSTM(sums,'ja')







