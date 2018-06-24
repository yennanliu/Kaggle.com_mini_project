
import pandas as pd, numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
%matplotlib inline
%pylab inline
import seaborn  as sns 
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression, LinearRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.grid_search import GridSearchCV





train_df = pd.read_csv("~/Desktop/titanic_train.csv", dtype={"Age": np.float64}, )
test_df  = pd.read_csv("~/Desktop/titanic_test.csv", dtype={"Age": np.float64}, )


x = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
print accuracy_score(x, train_df['Survived'])

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    fig.colorbar(im)

    target_names = ['not survived', 'survived']

    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()

df = train_df 
data =  df[['Age', 'Fare','Pclass', 'Survived']]
data['Age'] = data['Age'].fillna(data['Age'].mean())
#data = data[data.Age >0]

plt.figure()
sns.pairplot(data,hue="Survived", dropna=True)
plt.savefig("1_seaborn_pair_plot.png")

X = data[['Fare','Pclass', 'Age']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=4)


#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy on Testing Set: {:.3f}'.format(accuracy_score(y_test, y_test_pred)))
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
plot_confusion_matrix(cm)


#KNN - cross validation
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=33)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Accuracy on Testing Set: {:.3f}'.format(accuracy_score(y_test, y_test_pred)))



#kNN - check the evolution of accuracy  for different "neighbors"

k_range = range(1,50)
k_scores = []
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)


for k in k_range:

	

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
	k_scores.append(scores.mean())

print k_scores 

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')






