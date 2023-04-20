
import pandas as pd
import numpy as np

data_train = pd.read_csv('train.csv')
data_train

data_train.dtypes

data_train.isna().sum()

data_train.drop(['Name','Ticket','Fare','Cabin'], axis = 1, inplace = True)

data_train.Age.mean()
Age_mean = data_train.Age.mean() 
data_train.Age = data_train.Age.fillna(Age_mean)

data_train.Embarked = data_train.Embarked.fillna(0)

data_train.isna().sum()

data_train.Survived.value_counts()

data_train

for i in data_train.Age:
    if i < 18:
        data_train.Age = 'Child'
    else:
        data_train.Age = 'Adult'

data_train.Sex[data_train.Sex == 'male'] = 1
data_train.Sex[data_train.Sex == 'female'] = 0
data_train.Embarked[data_train.Embarked == 'S'] = 1
data_train.Embarked[data_train.Embarked == 'C'] = 2
data_train.Embarked[data_train.Embarked == 'Q'] = 3
data_train.Age[data_train.Age == 'Child'] = 0
data_train.Age[data_train.Age == 'Adult'] = 1

data_train


# ML

X = data_train.iloc[:,2:]
Y = data_train.iloc[:,1]

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits = 10)


# Log Reg
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
results = cross_val_score(LogReg, X, Y, cv = kfold)
print(results.mean())


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTree = DecisionTreeClassifier()
results = cross_val_score(DTree, X, Y, cv = kfold)
print(results.mean())

# Bagging
from sklearn.ensemble import BaggingClassifier
Bag = BaggingClassifier(base_estimator = DecisionTreeClassifier(), max_samples = 0.8, n_estimators = 100)
results = cross_val_score(Bag, X, Y, cv = kfold)
print(results.mean())

# Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100, max_features = 3)
results = cross_val_score(RF, X, Y, cv = kfold)
print(results.mean())

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
Ada = AdaBoostClassifier(n_estimators = 100)
results = cross_val_score(Ada, X, Y, cv = kfold)
print(results.mean())

# KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 17)
results = cross_val_score(KNN, X, Y, cv = kfold)
print(results.mean())

from sklearn.model_selection import GridSearchCV
model = KNeighborsClassifier()
n_neighbors = np.array(range(10,20))
grid = GridSearchCV(estimator = model, param_grid = dict(n_neighbors = n_neighbors), cv = 5)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_params_)

KNN = KNeighborsClassifier(n_neighbors = 15)
results = cross_val_score(KNN, X, Y, cv = kfold)
print(results.mean())

# SVC
from sklearn.svm import SVC
SV = SVC(kernel = 'rbf', gamma = 0.001)
results = cross_val_score(SV, X, Y, cv = kfold)
print(results.mean())

model = SVC()
param_grid = [{'kernel' : ['rbf'], 'gamma':[50,10,5,1,0.5,0.1,0.001,0.0001], 'C':[15,10,5,1,0.1]}]
gsv = GridSearchCV(model, param_grid, cv = 10)
gsv.fit(X, Y)
gsv.best_params_, gsv.best_score_

model = SVC(C = 1, gamma = 0.1, kernel = 'rbf')
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())

# Naive Bayes
from sklearn.naive_bayes import BernoulliNB
NB = BernoulliNB()
results = cross_val_score(NB, X, Y, cv = kfold)
print(results.mean())

# Final Model and Predictions
data_test = pd.read_csv('test.csv')
data_test
data_test.isna().sum()
data_test.drop(['Name','Ticket','Fare','Cabin'], axis = 1, inplace = True)
data_test.Age.mean()
Age_mean = data_test.Age.mean() 
data_test.Age = data_test.Age.fillna(Age_mean)
data_test.isna().sum()
data_test

for i in data_test.Age:
    if i < 18:
        data_test.Age = 'Child'
    else:
        data_test.Age = 'Adult'

data_test.Sex[data_test.Sex == 'male'] = 1
data_test.Sex[data_test.Sex == 'female'] = 0
data_test.Embarked[data_test.Embarked == 'S'] = 1
data_test.Embarked[data_test.Embarked == 'C'] = 2
data_test.Embarked[data_test.Embarked == 'Q'] = 3
data_test.Age[data_test.Age == 'Child'] = 0
data_test.Age[data_test.Age == 'Adult'] = 1

data_test
X_test = data_test.iloc[:,1:]

Final_model = SVC(C = 1, gamma = 0.1, kernel = 'rbf')
Final_model.fit(X, Y)

Y_pred = Final_model.predict(X_test)

Y_pred

ID = data_test.PassengerId

ID

Titanic_Pred = pd.DataFrame({'PassengerId': ID, 'Survived': Y_pred})
Titanic_Pred
Titanic_Pred.to_csv('Titanic_final.csv', index = False)
