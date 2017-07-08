
# basic library 
import pandas as pd, numpy as np
import pylab as pl
import pickle

# ml library 

############## data preparation  ##############

def load_data():
    df_train =  pd.read_csv('train.csv')
    df_test =  pd.read_csv('test.csv')
    return df_train, df_test


def feature_extract(df):
    # get frequency access with conditions in data
    df_ = df.copy()
    # cnt 
    df_['resource_cnt'] = df_.groupby('RESOURCE')['ACTION'].transform(len)
    df_['mgr_cnt'] = df_.groupby('MGR_ID')['ACTION'].transform(len)
    df_['role_cnt'] = df_.groupby('ROLE_CODE')['ACTION'].transform(len)
    # approve cnt 
    df_['approve_resource_cnt'] = df_.groupby('RESOURCE')['ACTION'].transform(sum)
    df_['approve_mgr_cnt'] = df_.groupby('MGR_ID')['ACTION'].transform(sum)
    df_['approve_role_cnt'] = df_.groupby('ROLE_CODE')['ACTION'].transform(sum)
    # pct 
    df_['resource_approve_pct'] = df_['approve_resource_cnt']/df_['resource_cnt']
    df_['mgr_approve_pct'] = df_['approve_mgr_cnt']/df_['mgr_cnt']
    df_['role_approve_pct'] = df_['approve_role_cnt']/df_['role_cnt']
    #print (df_.head(3))
    # filter no need features 
    select_feature = ['ACTION','RESOURCE', 'MGR_ID','ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY','ROLE_CODE','resource_cnt','mgr_cnt','role_cnt']
    df_ = df_[select_feature]
    return df_ 


def data_clean(df):
    # remove possible outliers 
    df_ = df.copy()
    df_ = df_[df.resource_cnt > df.resource_cnt.quantile(.5) ] 
    df_ = df_[df.mgr_cnt > df.mgr_cnt.quantile(.5) ] 
    df_ = df_[df.role_cnt > df.role_cnt.quantile(.5) ] 
    return df_ 


############## data process  ##############


def tuned_data():
    df_train, df_test = load_data()
    df_train_ = feature_extract(df_train)
    df_train_ = data_clean(df_train_)
    print (df_train_.head())
    return df_train_ 


#if __name__ == "__main__":
	#test_data_predict()
    #pass 
    #tuned_data()










