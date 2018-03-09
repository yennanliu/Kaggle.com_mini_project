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









if __name__ == '__main__':
	train = load_data()
	train['lang'] = train.Page.map(get_language)
	print (train.head(3))






