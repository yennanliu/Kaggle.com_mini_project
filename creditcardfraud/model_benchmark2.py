# python 3 


# import library 

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix



np.random.seed(7)

#-----------------------------------
# help function 

def get_data():
	df = pd.read_csv("data/creditcard.csv")
	return df


def get_train_test_data(df):
	print (df.head())
	print (df.columns)
	X = df.ix[:, df.columns != 'Class']
	Y = df.ix[:, df.columns != 'Class']


def get_under_sample_train_test_data(df):
	# get the number of  class = 1
	num_data_fraud = len(df[df.Class == 1 ])
	# get the index of data labeled as fraud (class = 1 )
	fraud_indices = np.array(df[df.Class == 1].index)
	# get the index of data labeled as fraud (class = 1 )
	normal_indices = np.array(df[df.Class == 1].index)
	# random select the data  
	###### can modify here ######
	random_normal_indices = np.random.choice(normal_indices, num_data_fraud, replace = False)
	random_normal_indices = np.array(random_normal_indices)
	# merge data together  (fraud_indices ,  random_normal_indices)
	under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
	# re-sample (Under Sample) here : num(class =1 ) = num(class =0)
	under_sample_data = df.iloc[under_sample_indices,:]
	X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
	y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
	print ('fraud under sample data : ', len(y_undersample[y_undersample.Class==1]) )
	print ('normal under sample data : ', len(y_undersample[y_undersample.Class==0]) )
	print ('under sample data : ', len(under_sample_data) )
	return under_sample_data, X_undersample, y_undersample


if __name__ == '__main__':
	df = get_data()
	# preprocess data 
	#X,Y = get_train_test_data(df)
	under_sample_data, X_undersample, y_undersample = get_under_sample_train_test_data(df)
	print (under_sample_data.head())
	print (X_undersample.head())




