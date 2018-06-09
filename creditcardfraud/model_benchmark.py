
# python 3 

# import library 

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix



np.random.seed(7)

#-----------------------------------


def get_data():
	df = pd.read_csv("data/creditcard.csv")
	return df 

def data_preprocess(df):
	# split train, dev, test 
	limit = int(0.9*len(df))
	train = df.loc[:limit]
	dev_test = df.loc[limit:]
	dev_test.reset_index(drop=True, inplace=True)
	dev_test_limit = int(0.5*len(dev_test))
	dev = dev_test.loc[:dev_test_limit]
	test = dev_test.loc[dev_test_limit:]
	return train, dev_test, dev, test 


def get_train_test_dev(train):
	# resample positive, negative instances for much balanced data points  
	train_positive = train[train["Class"] == 1]
	train_positive = pd.concat([train_positive] * int(len(train) / len(train_positive)), ignore_index=True)
	noise = np.random.uniform(0.9, 1.1, train_positive.shape)
	train_positive = train_positive.multiply(noise)
	train_positive["Class"] = 1
	train_extended = train.append(train_positive, ignore_index=True)
	train_shuffled = train_extended.sample(frac=1, random_state=0).reset_index(drop=True)
	#-----------------------------------
	X_train = train_shuffled.drop(labels=["Class"], axis=1)
	Y_train = train_shuffled["Class"]
	X_dev = dev.drop(labels=["Class"], axis=1)
	Y_dev = dev["Class"]
	X_test = test.drop(labels=["Class"], axis=1)
	Y_test = test["Class"]
	return X_train, Y_train, X_dev, Y_dev, X_test, Y_test



#-----------------------------------


if __name__ == '__main__':
	df = get_data()
	train, dev_test, dev, test  = data_preprocess(df)
	X_train, Y_train, X_dev, Y_dev, X_test, Y_test = get_train_test_dev(train)
	#### modeling ####
	lr_model = LogisticRegression(random_state=0).fit(X_train, Y_train)
	print("Train Accuracy:", lr_model.score(X_train, Y_train))
	print("Dev Accuracy:", lr_model.score(X_dev, Y_dev))
	lr_predict_train = lr_model.predict(X_train)
	lr_predict_dev = lr_model.predict(X_dev)
	print(average_precision_score(Y_train, lr_predict_train))
	print(average_precision_score(Y_dev, lr_predict_dev))










