		
# basic library 		
import pandas as pd, numpy as np		
import pylab as pl		
import pickle		
		
# ml library 		
from sklearn import svm		
		
		
		
def load_data():		
	df_test =  pd.read_csv('test.csv')		
	print (df_test.head())		
	return df_test		
		
def load_model():		
    with open('final_tuned_model.pkl', 'rb') as fid:		
        loaded_model = pickle.load(fid)		
        return loaded_model		
 		
		
def test_data_predict():		
    # submit prediction from TEST data 		
    df_test= load_data()		
    print (df_test.head())		
    # split train, test 		
    #X_train_, X_test_, y_train_, y_test_ = sample_split(df)		
    # load model 		
    model = load_model()		
    print (model)	
    #model = model(probability=True)
    # create predict dataframe 		
    df_predict=pd.DataFrame()		
    df_predict['Action'] = model.predict_proba(df_test.iloc[:,1:])[:, 1]		
    df_predict.index.name = 'ID'		
    # make index feat submission form 		
    # https://www.kaggle.com/c/amazoemployeacceschallenge/submit		
    df_predict.index = df_predict.index + 1		
    #print (df_predict.head())		
    df_predict.to_csv('df_predict_final.csv')		
    #df_read = pd.read_csv('df_predict_final.csv')		
    print (df_predict.Action.value_counts())		
    return df_predict		
		
		
if __name__ == "__main__":		
    test_data_predict()		




    
