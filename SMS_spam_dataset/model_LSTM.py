# python 3

"""
modify from  

https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

"""



# OP 
import pandas as pd
import numpy as np
import time

# ML 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import metrics



# ---------------------------
# help func 

# DL 
def LSTM_model():
	max_words = 1000
	max_len = 150
	inputs = Input(name='inputs',shape=[max_len])
	layer = Embedding(max_words,50,input_length=max_len)(inputs)
	layer = LSTM(64)(layer)
	layer = Dense(256,name='FC1')(layer)
	layer = Activation('relu')(layer)
	layer = Dropout(0.5)(layer)
	layer = Dense(1,name='out_layer')(layer)
	layer = Activation('sigmoid')(layer)
	model = Model(inputs=inputs,outputs=layer)
	return model


def LSTM_model_V2():
    max_len = 150
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=[max_len]))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc',metrics.binary_accuracy])
    print('compile done')
    return model

# ---------------------------
# main func 

def main():
	df = pd.read_csv('spam.csv', delimiter=',',encoding='latin-1')
	df = df.loc[:,['v1','v2']]
	df.head()
	# split train / test 
	print (' # ------------  split train / test  ------------ ')
	X = df.v2
	Y = df.v1
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
	# Process the data
	print (' # ------------  Process the data  ------------ ')
	max_words = 1000
	max_len = 150
	tok = Tokenizer(num_words=max_words)
	tok.fit_on_texts(X_train)
	sequences = tok.texts_to_sequences(X_train)
	sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
	# compile the model and print its architecture
	print (' # ------------  compile the model and print its architecture  ------------ ')
	#model = LSTM_model()
	model = LSTM_model_V2()
	model.summary()
	#model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
	# train on train set 
	print (' # ------------  train on train set   ------------ ')
	model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
	# predict on test set 
	print (' # ------------  predict on test set    ------------ ')
	test_sequences = tok.texts_to_sequences(X_test)
	test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
	accr = model.evaluate(test_sequences_matrix,Y_test)
	print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))








# ---------------------------
if __name__ == '__main__':
	main()





