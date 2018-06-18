# creditcardfraud


# Intro 

Identify fraudulent credit card transactions. (It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.)


# Metric 

- Accuracy = (TP+TN)/total
- Precision = TP/(TP+FP)
- Recall = TP/(TP+FN)
- ROC curve 
- Precision-Recall curve (AUPRC)


# Tech 
- python 3, Scikit-learn

# Quick Start

```bash

$ git pull https://github.com/yennanliu/Kaggle.com_mini_project.git
$ cd ~ && cd  Kaggle.com_mini_project/creditcardfraud/data  && brew install unzip && unzip creditcardfraud.zip 
$python model_RF.py
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





