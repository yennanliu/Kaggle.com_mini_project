# creditcardfraud


# Intro 

Identify fraudulent credit card transactions. (It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.)

# Theory (Fraud detection)

- Process 
	- step 1) data visualizartion : scan potential fraud relative data points, e.g. : fraud data VS normal data 
	- step 2) algorithm : 
	   - time dependent : e.g. credit card transaction : [100,120,150, 1000000, 100,103]
	      - time-series analysis 
	   - time independent : assume each transaction/fraud is "independent", so time-series will be removed in this case 
	      - unsupervised/supervised modeling 
    - step 3) validate fraud detection outcome with domain specialist
    - step 4) repeat step 1-3, keep optimize models and have logical phenomenon explanation 
    - step 5) deploy models to prod and track the performance 


- Metric
  - Confusion matrix : 
  ```python
  # given a confusion matrix :
   [[90  10]
   [ 10 90]]
  # The x-asis means "true group" ; while y-axis means "predict group"
  # i.e. [90  10] means 90 count of data is in group 0 and predicted as group 0 ; 10 count of data is in group 1 and predicted as group 0 
  # i.e. [10  90] means 10 count of data is in group 0 and predicted as group 1 ; 90 count of data is in group 1 and predicted as group 1
  # So, we can SAY : over all 200 (90+10+10+90) data sample, 
  # 90+90 = 180 data is predicted accurately 
  # 10 of [90  10] data is predicted wrongly as group 1 
  # 10 of [10  90] data is predicted wrongly as group 0  

  ```


  - Accuracy = (TP+TN)/total : 
  ```python 
  # Again, given confusion matrix as above :
  [[90  10]
  [ 10 90]]
  # The "OVERALL accuracy" of the model can be defined :
  # Accuracy = (TP+TN)/total , 
  # TP = true positive, data is in 1 group (true), predict as 1 group (true) 
  # TN = true negative, data is in 0 group (false), predict as 0 group (false) 
  # so we say the Accuracy = (90+90)/(90+10+10+90) = 90% is the model's "OVERALL" accuracy of all data points 

  ```
  - Precision = TP/(TP+FP)
  ```python 
  # Given confusion matrix 
  [[90  10]
  [ 10 90]]
  # "how accuracy the model predict `True` (predict) from all `True`(actual) group" 
  # can be defined as TP/(TP+FP) 
  # FP = false positive, data is in 0 group (negative), BUT predict as 1 group (true)
  # In this case, the Precision = (90)/(90+10) = 90%

  ```
  - Recall = TP/(TP+FN)
  ```python 
  # also known as "Sensitivity" or "Recall"
  # When the data is actually `True`(actual), how accurate the model  predict it is `True` ?
  # For the confusion matrix 
  [[90  10]
  [ 10 90]]
  # the Recall = (90)/(10+90) = 90%

  ```
  - ROC curve  (receiver operating characteristic)
  ```python
  # The curve with different thresholds, positive rate ( y-axis), and false positive rate (x-axis)
  # i.e. 
  # true_positive_rate, false_positive_rate = ROC(thresholds)
  # calculate the  true_positive_rate, false_positive_rate with "different thresholds", and plot all above as ROC curve
  # e.g. 
  # ------------
  >>> import numpy as np
  >>> from sklearn import metrics
  >>> y = np.array([1, 1, 2, 2])
  >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
  >>> fpr
  array([ 0. ,  0.5,  0.5,  1. ])
  >>> tpr
  array([ 0.5,  0.5,  1. ,  1. ])
  >>> thresholds
  array([ 0.8 ,  0.4 ,  0.35,  0.1 ])
  # ------------
  # ref 
  # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/ranking.py#L453
  # https://notmatthancock.github.io/2015/08/19/roc-curve-part-2-numerical-example.html

  ```
  - AUC (Area Under Curve)
  ```python 
  # total area under the ROC curve,
  # The model perform better when AUC -> 1 
  ```

  - Precision-Recall curve (AUPRC)


# Tech 
- python 3, Scikit-learn, numpy, pandas 

# Quick Start

```bash

$ git clone https://github.com/yennanliu/Kaggle.com_mini_project.git
$ cd ~ && cd  Kaggle.com_mini_project/creditcardfraud/data  && brew install unzip && unzip creditcardfraud.zip && cd .. 
$ python model_RF.py
# output

 X_undersample :  984
 y_undersample :  984
class : 
normal , fraud  :  (array([0, 1]), array([492, 492]))
---------------- grid search  ---------------- 
BEST {'max_features': 'auto', 'max_depth': 50, 'n_estimators': 50} 0.938953488372093 [mean: 0.93895, std: 0.02274, params: {'max_features': 'auto', 'max_depth': 50, 'n_estimators': 50}] <function _passthrough_scorer at 0x115c66620>
Best score:  0.938953488372093
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=50, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
---------------- confusion  matrix ----------------
[[143  16]
 [  4 133]]
---------------- ROC curve  ---------------- 

```
# Ref 
- https://www.kaggle.com/mlg-ulb/creditcardfraud
- http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/





