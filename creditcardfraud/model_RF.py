# import library 

import numpy as np
import pandas as pd
import os
from sklearn.metrics import average_precision_score, confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import itertools
#import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

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


def get_under_sample_train_test_data(X,y):
	X_undersample, y_undersample = RandomUnderSampler(random_state=10).fit_sample(X, y)
	print (' X_undersample : ' , len(X_undersample))
	print (' y_undersample : ' , len(y_undersample))
	print ('class : ', )
	print ('normal , fraud  : ', np.unique(y_undersample, return_counts=True))
	#print (' normal : ', len( y_undersample[y_undersample['Class'] == 0] ))
	#print (' fraud : ' ,len( y_undersample[y_undersample['Class'] == 1] ))
	return X_undersample, y_undersample


#-----------------------------------
# ML 

def get_best_param_Kfold(X,y):
	#fold = 
	pass 


def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None, verbose=0):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func, verbose=verbose)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds, verbose=verbose)
    gs.fit(X, y)
    print ("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_, gs.scorer_)
    print ("Best score: ", gs.best_score_)
    best_model = gs.best_estimator_
    return best_model



def RF_model(X,y,):
	model_RF = RandomForestClassifier(max_depth=2, random_state=0)
	model_RF.fit(X, y) 
	return model_RF










#-----------------------------------




if __name__ == '__main__':
	df, X, y  = get_data()
	print ('X :', X )
	print (' y :', y)
	# get unser sample data 
	X_undersample, y_undersample = get_under_sample_train_test_data(X,y)
	model_RF = RF_model(X_undersample, y_undersample)
	print (model_RF)
	# CV search
	# dev  
	# grid search 
	parameters = {  "n_estimators": [50],"max_features": ["auto"] ,"max_depth": [50]}
	"""
	neet to reshape y here :
	y -> np.array(y).reshape(len(y),) or np.ravel(y)

	model_RF.py:61: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
	model_RF.fit(X, y)

	"""
	best_model = cv_optimize(model_RF,parameters,X_undersample,np.ravel(y_undersample))
	print (best_model)
	# re-train with best model from grid search 
	model_RF = best_model.fit(X_undersample, y_undersample )
	y_pred = best_model.predict(X_undersample)
	cnf_matrix = confusion_matrix(y_pred,y_undersample)
	print (cnf_matrix)
	# train with best parameter again 








