# python 3 


# import library 

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
import itertools
import matplotlib.pylab as plt


np.random.seed(7)

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


#-----------------------------------
# ML 

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

#-----------------------------------



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
	# get best super-parameter in logicregression model 
	c_best = get_best_param_Kfold(X_train_undersample,y_train_undersample)
	print (c_best)
	# plot confusion matrix 
	# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
	# dataset
	lr = LogisticRegression(C = c_best, penalty = 'l1')
	lr.fit(X_train_undersample,y_train_undersample.values.ravel())
	y_pred_undersample = lr.predict(X_test_undersample.values)

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test_undersample,y_pred_undersample)
	print ('--------------- cnf_matrix :  ---------------')
	print (cnf_matrix)
	np.set_printoptions(precision=2)

	print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

	# Plot non-normalized confusion matrix
	class_names = [0,1]
	plt.figure()
	plot_confusion_matrix(cnf_matrix
	                      , classes=class_names
	                      , title='Confusion matrix')
	plt.show()






