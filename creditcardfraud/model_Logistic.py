import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
import itertools
import matplotlib.pylab as plt

np.random.seed(7)

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
    # get the number of  class = 1
    num_data_fraud = len(df[df.Class == 1 ])
    # get the index of data labeled as fraud (class = 1 )
    fraud_indices = np.array(df[df.Class == 1].index)
    # get the index of data labeled as fraud (class = 1 )
    normal_indices = np.array(df[df.Class == 0].index)
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_ROC_curve(X_test, y_test,y_pred,model):
    #lr = LogisticRegression(C = best_c, penalty = 'l1')
    #y_pred_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)
    #y_pred_undersample_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)
    y_pred_score = model.decision_function(X_test.values)
    fpr, tpr, thresholds = roc_curve(y_test.values.ravel(),y_pred_score)
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

def get_best_param_Kfold(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    # Different C parameters
    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')
    # fix here :  .idxmax() -> .values.argmax()
    # https://github.com/pandas-dev/pandas/issues/18021
    best_c = results_table.loc[results_table['Mean recall score'].values.argmax()]['C_parameter']
    
    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    
    return best_c

def logisticregression_model(X_train,y_train, X_test,y_test,c_param):
    lr_model = LogisticRegression(C = c_param, penalty = 'l1')
    lr_model.fit(X_train,y_train.values.ravel())
    y_test_pred = lr_model.predict(X_test.values)
    return y_test_pred, lr_model

if __name__ == '__main__':
    df, X, y  = get_data()
    print ('X :', X )
    print (' y :', y)
    # preprocess data 
    under_sample_data, X_undersample, y_undersample = get_under_sample_train_test_data(df)
    # get whole X, y data 
    X_train, X_test, y_train, y_test = get_train_test_data(X,y)
    # get undersample X, y data 
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = get_train_test_data(X_undersample, y_undersample)
    print ('################ Train With Undersample data ################')
    # get best super-parameter in logicregression model 
    c_best = get_best_param_Kfold(X_train_undersample,y_train_undersample)
    print (c_best)
    # re-train with best c parameter 
    y_pred_undersample,lr_model = logisticregression_model(X_train_undersample,y_train_undersample,X_test_undersample,y_test_undersample,c_best)
    cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
    print ('---------------- confusion  matrix ----------------')
    print (cnf_matrix)
    print ('---------------- ROC curve  ----------------')
    plot_ROC_curve(X_test_undersample, y_test_undersample ,y_pred_undersample,lr_model)
    
    print ('################ Train With Whole data ################')
    # get best super-parameter in logicregression model 
    c_best_ = get_best_param_Kfold(X_train,y_train)
    print (c_best_)
    # re-train with best c parameter 
    y_pred, lr_model = logisticregression_model(X_train,y_train,X_test,y_test,c_best_)
    cnf_matrix = confusion_matrix(y_test,y_pred)
    print ('---------------- confusion  matrix ----------------')
    print (cnf_matrix)
    print ('---------------- ROC curve  ----------------')
    plot_ROC_curve(X_test, y_test ,y_pred,lr_model)
