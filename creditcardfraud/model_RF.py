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
    best = gs.best_estimator_
    return best



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
	# CV search 
	parameters = {"n_estimators": [50],
              "max_features": ["auto"], # ["auto","sqrt","log2"]
              "max_depth": [50]}
    """
    neet to reshape y here :
    y -> np.array(y).reshape(len(y),)
    

    model_RF.py:61: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  	model_RF.fit(X, y)

    """
	best = cv_optimize(model_RF,parameters,X,np.array(y).reshape(len(y),))
	print (best)
	#def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None, verbose=0):
	y_precit = model_RF.predict(X)
	cnf_matrix = confusion_matrix(y_precit,y)
	print (cnf_matrix)






