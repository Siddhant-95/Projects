
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv', header = None, sep = ',\s')
df

df.shape
df.info()

names = ['Age', 'Job_type','Random','Education','Edu_num','Marital_Status','Occupation','Family_Role','Race','Gender','Gain','Loss','Hrsperweek','Country','Income']
df.columns = names

df.info()
df.Gain.value_counts()
df.Loss.value_counts()

df_cat = df.select_dtypes(include = 'object')
df_cat

for i in df_cat.columns:
    print(df_cat[i].value_counts())

df_cat.Job_type.value_counts()

for i in df_cat.columns:
    df_cat[i].replace('?', np.NaN, inplace = True)

df_cat.Job_type.value_counts()

df_cat.isna().sum()

df_cat['Job_type'].mode()

for i in df_cat.columns:
    df_cat[i].fillna(df_cat[i].mode()[0], inplace = True)

for i in df_cat.columns:
    print(df_cat[i].value_counts())

df_cat.isna().sum()

df_cat_features = df_cat.drop('Income', axis = 1) 

df_cat_features = pd.get_dummies(df_cat_features)
df_cat_features
df_cat_features.shape

df_num = df.select_dtypes(include = ('int64','float64'))
df_num

df_num.isna().sum()

features = pd.concat([df_cat_features, df_num], axis = 1) 
features

from sklearn.preprocessing import RobustScaler
cols = features.columns
scaler = RobustScaler()
scaled_features = scaler.fit_transform(features)

X = pd.DataFrame(scaled_features, columns = cols)

Y = df['Income']
Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 3)
X_train.shape, X_test.shape

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_pred, Y_test)

Y_pred_train = model.predict(X_train)

accuracy_score(Y_pred_train, Y_train)

#Accuracy for general model
Y_test.value_counts()

accuracy_general = (8131/(8131+2615))
accuracy_general

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred,Y_test)
cm

from sklearn.metrics import classification_report
cr = classification_report(Y_pred, Y_test)
print(cr)

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

accuracy = (TP + TN)/(TP+TN+FP+FN)
accuracy

Precision = TP/(TP+FP)
Precision

Recall = TP/(TP+FN)
Recall

F1_Score = (2*Precision*Recall)/(Recall + Precision)
F1_Score

Y_pred1 = gnb.predict_proba(X_test)[:,1]
Y_pred1

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred1, pos_label = '>50K')
auc = roc_auc_score(Y_test, Y_pred1)
plt.plot(fpr, tpr, color = 'red', label = 'GaussianNB(area = %0.3f)'%auc)
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc = 'best')
plt.show()
