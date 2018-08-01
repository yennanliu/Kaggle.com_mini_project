# python 3 

"""

Credit 

https://wayinone.github.io/SMS
https://www.kaggle.com/onlyshadow/spam-or-ham-7-machine-learning-tools-walk-through

""" 

# OP 
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download()
#%matplotlib inline
import string
#import matplotlib.pyplot as plt


# ML 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb


# ------------- help func  ------------- 
# HELP FUNC 


class stemmed_tfidf():
    def __init__(self,max_features=5000):
        self.ps = PorterStemmer()
        self.vc = TfidfVectorizer(analyzer='word',#{‘word’, ‘char’}  Whether the feature should be made of word or character n-grams
                             stop_words = 'english',
                             max_features = max_features)
    def tfidf(self,ListStr):
        '''
        return: sklearn.feature_extraction.text.TfidfVectorizer
        '''
        table = self.vc.fit_transform([self.stem_string(s) for s in ListStr])
        return table
    def stem_string(self,s):
        '''
        s:str, e.g. s = "Get strings with string. With. Punctuation?"
        ps: stemmer from nltk module
        return: bag of words.e.g. 'get string with string with punctuat'
        '''    
        s = re.sub(r'[^\w\s]',' ',s)# remove punctuation.
        tokens = word_tokenize(s) # list of words.
        #a = [w for w in tokens if not w in stopwords.words('english')]# remove common no meaning words
        return ' '.join([self.ps.stem(w) for w in tokens])# e.g. 'desks'->'desk'


# ------------- help func  ------------- 


# ------------- main func  ------------- 
def main():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.loc[:,['v1','v2']]
    print (df.tail(5))
    d={'spam':1,'ham':0}
    df.v1 = list(map(lambda x:d[x],df.v1))
    stf = stemmed_tfidf()
    feature = stf.tfidf(df.v2) # this will be a sparse matrix of size (n,5000)
    print('%2.2f percent of data is spam: We have an inbalanced data set.'%round(100*sum(df.v1)/len(df),2))
    Xtrain, Xtest, ytrain, ytest = train_test_split(feature, df.v1, test_size=0.2, random_state=1)
    Acc = {}
    F1score = {}
    confusion_mat={}
    predictions = {}
    # ---- ML ----
    # ---------------- 1) SVM  ----------------
    print (' # 1) SVM  ')
    val_scores = []
    listc = np.linspace(0.5,3,num=4)
    listgamma = np.linspace(0.5,3,num=4)
    kernel = ['rbf','sigmoid']# 'poly' is doing bad here, let's save some time.
    for v in kernel:
        for c in listc:
            for gamma in listgamma:
                svc = SVC(kernel=v, C=c, gamma=gamma,class_weight='balanced')
                #3. The “balanced” mode uses the values of y to automatically adjust weights inversely 
                #   proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
                scores = cross_val_score(svc, Xtrain, ytrain,scoring='f1')
                val_scores.append([np.mean(scores),v, c,gamma])

    val_scores = np.array(val_scores)
    print('The best scores happens on:',val_scores[val_scores[:,0]==max(val_scores[:,0]),1:],
          ', where F1 =',val_scores[val_scores[:,0]==max(val_scores[:,0]),0])
    val_scores = []
    listc = np.linspace(0.5,2,num=5)
    listgamma = np.linspace(0.3,1,num=5)
    for c in listc:
        for gamma in listgamma:
            svc = SVC(kernel='sigmoid', C=c, gamma=gamma,class_weight='balanced')
            scores = cross_val_score(svc, Xtrain, ytrain,scoring='f1')
            val_scores.append([np.mean(scores),v, c,gamma])
    val_scores = np.array(val_scores)
    print('The best scores happens on:',val_scores[val_scores[:,0]==max(val_scores[:,0]),1:],
          ', where F1 =',val_scores[val_scores[:,0]==max(val_scores[:,0]),0])
    name = 'SVM'
    svc = SVC(kernel='sigmoid', C=1.25, gamma=0.825,class_weight='balanced')
    svc.fit(Xtrain,ytrain)
    pred = svc.predict(Xtest.toarray())
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))
    # ----------------  2) Naive Bayes ----------------
    print (' # 2) Naive Bayes ')
    from sklearn.naive_bayes import MultinomialNB
    val_scores = []
    listalpha = np.linspace(0.01,1,num=20)
    for i in listalpha:
        MNB = MultinomialNB(alpha=i)# alpha is Laplace smoothing parameter
        scores = cross_val_score(MNB, Xtrain, ytrain,scoring='f1')
        val_scores.append([np.mean(scores),i])
    val_scores = np.array(val_scores)
    print('The best scores happens on:',val_scores[val_scores[:,0]==max(val_scores[:,0]),1:],
          ', where F1 =',val_scores[val_scores[:,0]==max(val_scores[:,0]),0])
    name = 'MNB'
    MNB = MultinomialNB(alpha=0.27052632)
    MNB.fit(Xtrain,ytrain)
    pred = MNB.predict(Xtest.toarray())
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))
    # ----------------  3) Decision Tree ----------------
    print (' # 3) Decision Tree ')
    val_scores = []
    for i in range(2,21):
        DT = DecisionTreeClassifier(min_samples_split=i, random_state=1,class_weight='balanced')
        scores = cross_val_score(DT, Xtrain, ytrain,scoring='f1')
        val_scores.append([np.mean(scores),i])
    val_scores = np.array(val_scores)
    print('The best scores happens on:',val_scores[val_scores[:,0]==max(val_scores[:,0]),1:],
    ', where F1 =',val_scores[val_scores[:,0]==max(val_scores[:,0]),0])
    name = 'DT'
    bset_samples_split = int(val_scores[val_scores[:,0]==max(val_scores[:,0]),1:])
    DT = DecisionTreeClassifier(min_samples_split=bset_samples_split, random_state=1,class_weight='balanced')
    DT.fit(Xtrain,ytrain)
    pred = DT.predict(Xtest.toarray())
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))
    # ----------------  4) Logistic Regression ----------------
    print (' # 4) Logistic Regression ')
    list_C = np.linspace(0.1,3,num=20)
    val_scores = []
    for p in ('l1','l2'):
        for c in list_C:
            LR = LogisticRegression(penalty=p,C=c,class_weight='balanced')
            scores = cross_val_score(LR, Xtrain, ytrain,scoring='f1')
            val_scores.append([np.mean(scores),p,c])
    val_scores = np.array(val_scores)
    print('The best scores happens on:',val_scores[val_scores[:,0]==max(val_scores[:,0]),1:],
    ', where F1 =',val_scores[val_scores[:,0]==max(val_scores[:,0]),0])   
    name = 'LR'
    best_l2_c = float(val_scores[val_scores[:,0]==max(val_scores[:,0]),1:][0][1])
    LR = LogisticRegression(penalty='l2',C=best_l2_c,class_weight='balanced')
    LR.fit(Xtrain,ytrain)
    pred = LR.predict(Xtest)
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))
    # ----------------  5)  Random Forest ----------------
    print (' # 5) Random Forest ')  
    name = 'RF'
    RF = RandomForestClassifier(n_estimators =80,class_weight ='balanced')
    RF.fit(Xtrain,ytrain)
    pred = RF.predict(Xtest)
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))  
    # ----------------  6)  AdaBoost ----------------
    print (' # 6) AdaBoost ')  
    name = 'AdaBoost'
    Ada = AdaBoostClassifier(n_estimators=80, random_state=1)
    Ada.fit(Xtrain,ytrain)
    pred = Ada.predict(Xtest)
    F1score[name]= f1_score(ytest,pred)
    Acc[name] = accuracy_score(ytest,pred)
    confusion_mat[name] = confusion_matrix(ytest,pred)
    predictions[name]=pred
    print(name+': Accuracy=%1.3f, F1=%1.3f'%(Acc[name],F1score[name]))  
    # ----------------  7)  XGBoost ----------------
    print (' # 7) XGBoost ')   
    #  XGBoost, a faster and more efficient version of GBM.
    data_tr  = xgb.DMatrix(Xtrain, label=ytrain)
    val_scores = []
    list_max_depth =[6,9,14]
    list_subsample = [0.8,1]
    for max_depth in list_max_depth:
        for subsample in list_subsample:
            parms = {'max_depth':max_depth, #maximum depth of a tree
                     'objective':'binary:logistic',
                     'eta'      :0.1,
                     'subsample':subsample,#SGD will use this percentage of data
                     'lambda '  :1.5, #L2 regularization term,>1 more conservative
                     'colsample_bytree ':0.8,
                     'nthread'  :3}  #number of cpu core to use
            result = xgb.cv(parms, data_tr, 
                                num_boost_round=1000,
                                early_stopping_rounds=20,# early stop if cv result is not improving
                                nfold=3,metrics="error")
            val_scores.append([result['test-error-mean'].iloc[-1],max_depth,subsample,len(result)-20])
            #len(result) will be our num_boot_round in the test set
    val_scores = np.array(val_scores)
    print('The best scores happens on:',val_scores[val_scores[:,0]==min(val_scores[:,0]),1:],
          ', where accuracy =',val_scores[val_scores[:,0]==min(val_scores[:,0]),0])  





# ------------- run the process  ------------- 

if __name__ == '__main__':
    main()








