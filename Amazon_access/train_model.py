

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
from sklearn import metrics, cross_validation, preprocessing

# ml model 

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, linear_model


#############################

# help function 


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
    # model  score 
    model_score = model.score(X_test,y_test)
    print ('model  score : %.2f' %model_score)
    # confusion metrix 
    y_test_predict = model.predict(X_test)
    print (confusion_matrix(y_test_predict,y_test))
    return model


def save_results(preds, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(preds):
            f.write("%d,%f\n" % (i + 1, pred))


#############################

# modeling 

def feature_extract():
    df_train, df_test = load_data()
    X = np.array(df_train.iloc[:,1:])
    y = np.array(df_train.iloc[:,:1])
    X_test= np.array(df_test.iloc[:,1:])
    # === one-hot encoding === #
    # encode the category IDs encountered both in train and test set
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((X, X_test)))
    X = encoder.transform(X)  
    X_test = encoder.transform(X_test)
    return X,y,X_test



def model_fit():
    # === set features as labels === #
    X,y,X_test = feature_extract()
    SEED = 42
    mean_auc = 0.0
    #model = linear_model.LogisticRegression(C=3)
    model = linear_model.LogisticRegression()
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20, random_state=i*SEED)
        # train model and make predictions
        model.fit(X_train, y_train) 
        preds = model.predict_proba(X_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print ("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))
        # confusion matrix
        y_cv_predict = model.predict(X_cv)
        print (confusion_matrix(y_cv_predict,y_cv))
        mean_auc += roc_auc
    print (model)
    return model 


def predict():
    X,y,X_test = feature_extract()
    model = model_fit()
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]
    #filename = raw_input("Enter name for submission file: ")
    filename = "FINAL_SIBMIT_PRDICT"
    print (preds)
    save_results(preds, filename + ".csv")




#############################


if __name__ == "__main__":
    predict()







