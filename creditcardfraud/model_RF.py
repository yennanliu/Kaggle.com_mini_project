# import library 

import numpy as np
import pandas as pd
import os
from sklearn.metrics import average_precision_score, confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import itertools
#import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier

#-----------------------------------
# help function 

def get_data():
	df = pd.read_csv("data/creditcard.csv")
	print (df.head())
	print (df.columns)
	X = df.ix[:, df.columns != 'Class']
	y = df.ix[:, df.columns == 'Class']
	return df, X, y 


def get_train_test_data(X,y):
	# Whole dataset
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
	return X_train, X_test, y_train, y_test


def get_under_sample_train_test_data(df):
	pass 



#-----------------------------------
# ML 

def get_best_param_Kfold():
	pass 


def RF_model(X,y):
	model_RF = RandomForestClassifier(max_depth=2, random_state=0)
	model_RF.fit(X, y) 
	return model_RF










#-----------------------------------




if __name__ == '__main__':
	df, X, y  = get_data()
	print ('X :', X )
	print (' y :', y)
	# get best RF super paramter 
	# fit with best super paramter 
	model_RF = RF_model(X,y)
	print (model_RF)




