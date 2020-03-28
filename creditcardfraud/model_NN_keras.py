import numpy as np
import pandas as pd
import os

# OP 
import itertools
import matplotlib.pylab as plt

# ML 
from sklearn.metrics import average_precision_score, confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

# DL 
from keras.models import Sequential
from keras.layers import Dense

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

def NN_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def NN_small():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def NN_large():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main(build_fn):
    df, X, y  = get_data()
    print ('X :', X )
    print (' y :', y)
    # get under sample train/test data 
    X_undersample, y_undersample = get_under_sample_data(X,y)
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = get_train_test_data(X_undersample, y_undersample)
    ####  DL #### 
    seed = 7
    np.random.seed(seed)
    # https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    estimators = []
    # evaluate model with standardized dataset
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=build_fn, epochs=200, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(pipeline, X_train_undersample, y_train_undersample, cv=kfold)
    print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    print (results)
    #return results

if __name__ == '__main__':
    main(NN_large)
