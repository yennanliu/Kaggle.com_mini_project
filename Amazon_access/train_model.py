

# Load basics library 

import pandas as pd, numpy as np
import pylab as pl
import pickle
# import data prepare module 
from data_prepare import *

# ml 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


#############################

# working function 

def sample_split(data):
    #data =  data[selected_feature]
    relevent_cols = list(data)
    data_=data.values.astype(float)             
    Y = data_[:,0]
    X = data_[:,1:]
    test_size = .3
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = 3)
    return X_train, X_test, y_train, y_test

def reg_analysis(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    #Calculate Variance score
    Variance_score = explained_variance_score(y_test, prediction)
    print ('Variance score : %.2f' %Variance_score)
    #Mean Absolute Error
    MAE = mean_absolute_error(y_test, prediction)
    print ('Mean Absolute Error : %.2f' %MAE)
    #Root Mean Squared Error
    RMSE = mean_squared_error(y_test, prediction)**0.5
    print ('Mean Squared Error : %.2f' %RMSE)
    #R² score, the coefficient of determination
    r2s = r2_score(y_test, prediction)
    print ('R2  score : %.2f' %r2s)
    model_score = model.score(X_test,y_test)
    print ('model  score : %.2f' %model_score)
    # confusion metrix 
    y_test_predict = model.predict(X_test)
    print (confusion_matrix(y_test_predict,y_test))
    return model


#############################

# get tuned data 
df_train = tuned_data()
# split train, test set 
X_train, X_test, y_train, y_test = sample_split(df_train)

# Random forest 
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(random_state=0)
clf_forest_ = reg_analysis(clf_forest,X_train, X_test, y_train, y_test)













