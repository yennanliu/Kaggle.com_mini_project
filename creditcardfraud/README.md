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
				- unsurprised/surprised 
    - step 3) validate fraud detection outcome with domain specialist
    - step 4) repeat step 1-3, keep optimize models and have logical phenomenon explanation 
    - step 5) deploy models to prod and track the performance 



# Metric 

- ML classification 
	- Accuracy = (TP+TN)/total
	- Precision = TP/(TP+FP)
	- Recall = TP/(TP+FN)
	- ROC curve 
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





