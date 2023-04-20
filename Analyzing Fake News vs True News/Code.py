# Fake and True News
# The Objective is to use Machine Learning to create a model that can identify Fake and True News.
# In order to make the data suitable for Machine Learning, we must use NLP and EDA to extract features from the textual Data.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fake news datframe
fake_df = pd.read_csv('fake.csv')
fake_df

# True news datframe
true_df = pd.read_csv('true.csv')
true_df

# Adding clss column to the dataframe which will be our target variable
fake_df['class'] = 0
true_df['class'] = 1

# Checking shape of the dtaframes
fake_df.shape, true_df.shape
# The column numbers are the same and the no of rows is also very close

# Combining both dataframes to create our final datset
df = pd.concat([fake_df, true_df])
df

# Randomizing all the rows and resetting the index so that the classes are not concentrated in upper or lower half.
df_news = df.sample(frac = 1).reset_index(drop = True)
df_news

# Dropping unnecessary columns
df_final = df_news.drop(['title','subject','date'], axis = 1)
df_final

# Final shape of the Datset
df_final.shape

# Checking the values of target variable
c = df_final['class'].value_counts()
p = df_final['class'].value_counts(normalize = True)
pd.concat([c,p], axis = 1, keys = ['count', '% count'])
# Balanced Dataset


# Natural Language Processing

# importing libraries for NLP
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Preprocessing function
def preprocessing(x):
        x = x.lower()
        x = re.sub(r'http\S+', '',x)
        x = re.sub('@[^\s]+','',x)
        x = re.sub('[^a-zA-Z0-9]'," ",x)
        return x

df_final.text = df_final.text.apply(preprocessing)

df_final

df_final.text = df_final.text.apply(lambda x : x.split())

sw = stopwords.words('english')
print(len(sw))

# Removing Stopwords
df_final.text = df_final.text.apply(lambda x : [word for word in x if word not in sw])

df_final

# Using Porter Stemmer to get to root words
ps = PorterStemmer()
df_final.text = df_final.text.apply(lambda x: [ps.stem(word) for word in x])

df_final.text = df_final.text.apply(lambda x: " ".join(x))
df_final


# Train Test Split

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_final['text'], df_final['class'], test_size = 0.3, random_state = 2)

x_train

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# Tfidf Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

# Using Tfidf to extract features using the training dataset
tv = TfidfVectorizer()
xtv_train = tv.fit_transform(x_train)

xtv_train.shape

y_test.value_counts()

# Transforming the test data as per the the features extracted from the training data
xtv_test = tv.transform(x_test)

xtv_test.shape


# Machine Learning

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score


# Logistic Regression

lr = LogisticRegression()
lr.fit(xtv_train, y_train)

y_pred_lr = lr.predict(xtv_test) 

accuracy_score(y_pred_lr, y_test)

print(classification_report(y_test, y_pred_lr))


# Decision Tree Classifier

dt = DecisionTreeClassifier()
dt.fit(xtv_train, y_train)

y_pred_dt = dt.predict(xtv_test)

accuracy_score(y_test, y_pred_dt)

print(classification_report(y_test, y_pred_dt))


# Random Forest Classifier

rf = RandomForestClassifier(n_estimators = 100, max_features = 100)
rf.fit(xtv_train,y_train)

y_pred_rf = rf.predict(xtv_test)

accuracy_score(y_test, y_pred_rf)

print(classification_report(y_test, y_pred_rf))


# eXtreme Gradient Boosting

xgb = XGBClassifier(n_estimators = 100)
xgb.fit(xtv_train, y_train)

y_pred_xgb = xgb.predict(xtv_test)

accuracy_score(y_test, y_pred_xgb)

print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix of all the models
from sklearn import metrics
model = [lr, dt, rf, xgb]
for i in model:
    print(i)
    cm_i = metrics.confusion_matrix(y_test, i.predict(xtv_test))
    cm_display_i = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_i, display_labels = [False, True])
    cm_display_i.plot()
    plt.show()

# Classification Report for all models
for i in model:
    print(i)
    print(classification_report(y_test, i.predict(xtv_test)))


# Conclusion
# 1. Decision Tree Classifier and Extreme Gradient Boost have the best metrics.
# 2. However XGB has less number of misclassifications.
# 2. Hence we can go with XGB as our model for fake news detection.
