# import library 

# op 
import numpy as np
import pandas as pd
import os
import itertools

#import matplotlib.pylab as plt
# ML 
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier,GradientBoostingClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, make_scorer,average_precision_score, confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cross_validation import train_test_split, KFold, cross_val_score


#-----------------------------------
# help function 

def get_data():
	df = pd.read_csv("data/creditcard.csv")
	#print (df.head())
	#print (df.columns)
	X = df.ix[:, df.columns != 'Class']
	y = df.ix[:, df.columns == 'Class']
	return df, X, y 


def get_train_test_data(X,y):
	# Whole dataset
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
	return X_train, X_test, y_train, y_test


def get_under_sample_data(X,y):
	X_undersample, y_undersample = RandomUnderSampler(random_state=10).fit_sample(X, y)
	print (' X_undersample : ' , len(X_undersample))
	print (' y_undersample : ' , len(y_undersample))
	print ('class : ', )
	print ('normal , fraud  : ', np.unique(y_undersample, return_counts=True))
	#print (' normal : ', len( y_undersample[y_undersample['Class'] == 0] ))
	#print (' fraud : ' ,len( y_undersample[y_undersample['Class'] == 1] ))
	return X_undersample, y_undersample



def get_under_sample_train_test_data(X,y):
	pass 




def plot_ROC_curve_updated(X_test, y_test,y_pred,model):
	try:
		y_pred_score = model.decision_function(X_test.values)
	except:
		y_pred_score = model.predict_proba(X_test)
	# http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html
	fpr, tpr, thresholds = roc_curve(y_test,y_pred_score[:, 1])
	roc_auc = auc(fpr,tpr)
	# Plot ROC
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.0])
	plt.ylim([-0.1,1.01])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show() 


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


def cv_optimize_RF(clf, X, y, n_jobs=1, n_folds=10, score_func=None, verbose=0):
	param_grid = {'min_samples_split': [3, 5, 10], 
				  'n_estimators' : [100, 300],
				  'max_depth': [3, 5, 15, 25],
				  'max_features': [3, 5, 10, 20]}
	gs = GridSearchCV(clf, param_grid=param_grid, n_jobs=n_jobs, cv=n_folds, verbose=verbose)
	gs.fit(X, y)
	print ("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_, gs.scorer_)
	print ("Best score: ", gs.best_score_)
	best_model = gs.best_estimator_
	return best_model

def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train.values, y_train.values)

    # make the predictions
    y_pred = grid_search.predict(X_test.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

def RF_model(X,y):
	model_RF = RandomForestClassifier(max_depth=2, random_state=0)
	model_RF.fit(X, y) 
	return model_RF


def training_model(X,y,model_):
	model_ = model_()
	model_.fit(X, y) 
	return model_




#-----------------------------------
# main running func 
def main(model_,model_name):
	df, X, y  = get_data()
	#print ('X :', X )
	#print (' y :', y)
	# get under sample train/test data 
	X_undersample, y_undersample = get_under_sample_data(X,y)
	X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = get_train_test_data(X_undersample, y_undersample)
	model_ = training_model(X_undersample, y_undersample,model_)
	print (model_) 
	# grid search 
	print ('---------------- grid search  ---------------- ')
	print ('model_ : ', str(model_name))
	####### will fix the grid tuning here for all algorithms #######
	#parameters = {  "n_estimators": [50],"max_features": ["auto"] ,"max_depth": [50]}
	#parameters = {  "n_estimators": [50],"max_features": ["auto"]}
	if str(model_name)=="RF":
		print ('start CV optimize....')
		best_model = cv_optimize_RF(model_,X_train_undersample,np.ravel(y_train_undersample))
	else:
		print ('no CV optimize....')
		best_model = model_

	#print (best_model)
	# re-train with best model from grid search 
	#best_model = model_ 

	model_RF = best_model.fit(X_train_undersample, y_train_undersample )
	y_test_pred = best_model.predict(X_test_undersample)
	print ('---------------- confusion  matrix ----------------')
	cnf_matrix = confusion_matrix(y_test_pred,y_test_undersample)
	print (cnf_matrix)
	print ('---------------- classification report  ----------------')
	target_names=['0', '1']
	print (classification_report(y_test_undersample, y_test_pred,target_names=target_names))
	print ('---------------- ROC curve  ----------------')
	#plot_ROC_curve_updated(X_test_undersample, y_test_undersample ,y_test_pred,best_model)
	return cnf_matrix






#-----------------------------------




if __name__ == '__main__':
	models = []
	results= []
	names = []
	models.append(('LR', LogisticRegression))
	models.append(('LDA', LinearDiscriminantAnalysis))
	models.append(('KNN', KNeighborsClassifier))
	models.append(('CART', DecisionTreeClassifier))
	models.append(('NB', GaussianNB))
	models.append(('SVM', SVC))
	models.append(('RF', RandomForestClassifier))
	models.append(('GradientBoosting', GradientBoostingClassifier))

	for name,model in models:
		print ('model name :', name)
		cnf_matrix = main(model,name)
		results.append(cnf_matrix)
		names.append(name)
	for names,results in zip(names,results):
		print ('* model name :', names )
		print ('* accuracy :')
		print (results)





