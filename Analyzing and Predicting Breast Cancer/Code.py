# Breast Cancer Analysis
# The objective is to analyse the breast cancer dataset and to use Machine Learning in order to generate a predictve algorithm which can classify the tumor as Malignant "M" or Beningn "B"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df = pd.read_csv('data.csv')
df

df.info()

df = df.drop(['Unnamed: 32','id'], axis = 1) # Dropping unnecesary columns

df.shape

df[df.duplicated()]
# No duplicate values

# Checking the values of target variable
c = df.diagnosis.value_counts()
p = df.diagnosis.value_counts(normalize = True)
pd.concat([c,p], axis = 1, keys = ['count', '% count'])
# Unbalanced Dataset, so accuracy alone won't be a suitable metric for evaluation

df.corr() #Correlation check

plt.figure(figsize = (10,7))
sns.heatmap(df.corr(), center = 0, cmap = 'Blues')
# Some features have high correlation


# Seperation of features and target variable
features = df.drop(['diagnosis'], axis = 1)
target = df.diagnosis


# Splitting into training and testing
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 23) 


# Scaling the dataset using Standard Scaler(Standardisation)
from sklearn.preprocessing import StandardScaler
features_train_array = features_train.values
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train_array)


# Transforming the test data as per the trainig data
features_test_array = features_test.values
features_test_scaled = scaler.transform(features_test_array)


# Converting scaled values to dataframes
features_train_df = pd.DataFrame(features_train_scaled, columns = features_train.columns)
features_test_df = pd.DataFrame(features_test_scaled, columns = features_test.columns)


features_train_df.shape, features_test.shape

target_test.value_counts()


# Using PCA for feature reduction
from sklearn.decomposition import PCA
pca = PCA()
features_train_pca = pca.fit_transform(features_train_scaled)

features_train_pca

var = pca.explained_variance_ratio_
var


#Cummulative sum of variances of new features
cum_sum = np.cumsum(var)
cum_sum
# 2 featues represent 62.8% of variance, 10 features represent 95.1% of variance, 17 features represent 99.12% of variance


# Plotting the cummulative sum of the features
plt.figure(figsize = (7,5))
plt.bar(range(1,len(var) + 1), var, label = 'Individual')
plt.step(range(1,len(cum_sum) + 1), cum_sum, label = 'Combined')
plt.yticks(np.arange(0,1.1,0.1))
plt.legend(loc = 'best')
plt.show()


# pca values to dataframe
pca_train = features_train_pca[:,0:17]
pca_train_df = pd.DataFrame(pca_train,columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11',
                                                 'pc12','pc13','pc14','pc15','pc16','pc17'])


pca_train_df


# Using the first two features to plot the values
pca_2 = pca_train_df[['pc1','pc2']]
pca_d = pd.DataFrame(data = target_train.values, columns = ['diagnosis'])
pca_plot = pd.concat([pca_2, pca_d], axis = 1)


sns.lmplot(x = 'pc1', y = 'pc2', data = pca_plot, hue = 'diagnosis', fit_reg = False, legend = True )

# We can see the different classes visually
# even though these two features contain 63% of variance, this gives us a descent idea about how the classes differ


sns.histplot(pca_plot[pca_plot.diagnosis == "B"].iloc[:,0], kde = True, color= "green")
sns.histplot(pca_plot[pca_plot.diagnosis == "M"].iloc[:,0], kde = True, color = "red")
plt.show()

# We can see the spread of the data in case of pc1 which gives us 43% of the variance
# The peaks of the classes are different although there is slight overlap


# Transforming the test data as per the PCA obtained from training data
features_test_pca = pca.transform(features_test_scaled)
pca_test = features_test_pca[:,0:17]
pca_test_df = pd.DataFrame(pca_test,columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11',
                                                 'pc12','pc13','pc14','pc15','pc16','pc17'])


pca_test_df


# Machine Learning

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Logistic Regression

# Using Standardized DataFrame
LogReg = LogisticRegression()
LogReg.fit(features_train_df, target_train)

pred_lr = LogReg.predict(features_test_df)

accuracy_score(target_test, pred_lr)

confusion_matrix(target_test, pred_lr)

print(classification_report(target_test, pred_lr))


# Using PCA DataFrame
LogReg.fit(pca_train_df, target_train)

pred_lr_pca = LogReg.predict(pca_test_df)

accuracy_score(target_test, pred_lr_pca)

confusion_matrix(target_test, pred_lr_pca)

print(classification_report(target_test, pred_lr_pca))

# High accuracy, precision, recall and f1 score using Logistic Regression.
# Accuracy, precision, recall and f1 score for scaled dataframe and pca dataframe are the same.


# Decision Tree Classifier

# Using Standardized DataFrame
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_features = 7)
DT.fit(features_train_df, target_train)

pred_dt = DT.predict(features_test_df)

accuracy_score(target_test, pred_dt)

confusion_matrix(target_test, pred_dt)

print(classification_report(target_test, pred_dt))


# Using PCA DataFrame
DT.fit(pca_train_df, target_train)

pred_dt_pca = DT.predict(pca_test_df)

accuracy_score(target_test, pred_dt_pca)

confusion_matrix(target_test, pred_dt_pca)

print(classification_report(target_test, pred_dt_pca))

# Significant drop in all the metrics when using Decision Tree Classifier compared to Logistic Regression


# Random Forest Classifier

# Using Standardized DataFrame
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100, max_features = 7, random_state = 7)
RF.fit(features_train_df, target_train)

pred_rf = RF.predict(features_test_df)

accuracy_score(target_test, pred_rf)

confusion_matrix(target_test, pred_rf)

print(classification_report(target_test, pred_rf))


# Using PCA DataFrame
RF.fit(pca_train_df, target_train)

pred_rf_pca = RF.predict(pca_test_df)

accuracy_score(target_test, pred_rf_pca)

confusion_matrix(target_test, pred_rf_pca)

print(classification_report(target_test, pred_rf_pca))

# Increase in accuracy when using RF and PCA compared to Logistic Regression


# Bagging Classifier

# Using Standardized DataFrame
from sklearn.ensemble import BaggingClassifier
Bag = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 100, max_samples = 0.8, random_state = 8)
Bag.fit(features_train_df, target_train)

pred_bag = Bag.predict(features_test_df)

accuracy_score(target_test, pred_bag)

confusion_matrix(target_test, pred_bag)

print(classification_report(target_test, pred_bag))


# Using PCA DataFrame
Bag.fit(pca_train_df, target_train)

pred_bag_pca = Bag.predict(pca_test_df)

accuracy_score(target_test, pred_bag_pca)

confusion_matrix(target_test, pred_bag_pca)

print(classification_report(pred_bag_pca, target_test))

# Random Forest still the best model yet.


# KNN

# Using Standardized data
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(features_train_df, target_train)

pred_knn = KNN.predict(features_test_df)

accuracy_score(target_test, pred_knn)

#GridSearch for best 'K'
from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,20))
par = dict(n_neighbors = n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator = model, param_grid = par)
grid.fit(features_train_df, target_train)

print(grid.best_score_)
print(grid.best_params_)

# Using best 'K' i.e. 4
KNN_best = KNeighborsClassifier(n_neighbors = 4)
KNN_best.fit(features_train_df, target_train)

pred_knn_best = KNN_best.predict(features_test_df)

accuracy_score(target_test, pred_knn_best)

confusion_matrix(target_test, pred_knn_best)

print(classification_report(target_test, pred_knn_best))


# Using PCA DAtaFrame
KNN_best.fit(pca_train_df, target_train)

pred_knn_pca = KNN_best.predict(pca_test_df)

accuracy_score(target_test, pred_knn_pca)

confusion_matrix(target_test, pred_knn_pca)

print(classification_report(target_test, pred_knn_pca))

# Descent model for the dataset, however, Logistic Regression still slightly better and Random Forest with PCA the best


# SVM using Radial Kernel

# Using Standardized DataFrame
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', gamma = 0.001)
svc.fit(features_train_df, target_train)

pred_svc = svc.predict(features_test_df)

accuracy_score(target_test, pred_svc)

# Grid Search for best value of 'gamma'
model = SVC()
par = [{'kernel': ['rbf'], 'gamma' : [5,1,0.01,0.001,0.0001]}]
grid = GridSearchCV(estimator = model, param_grid = par)
grid.fit(features_train_df, target_train)

print(grid.best_score_)
print(grid.best_params_)

#Using best 'gamma' i.e. 0.01
svc_best = SVC(kernel = 'rbf', gamma = 0.01)
svc_best.fit(features_train_df, target_train)

pred_svc_best = svc_best.predict(features_test_df)

accuracy_score(target_test, pred_svc_best)

confusion_matrix(target_test, pred_svc_best)

print(classification_report(target_test, pred_svc_best))


# Using PCA DataFrame
svc_best.fit(pca_train_df, target_train)

pred_svc_pca = svc_best.predict(pca_test_df)

accuracy_score(target_test, pred_svc_pca)

confusion_matrix(target_test, pred_svc_pca)

print(classification_report(target_test, pred_svc_pca))

# The metrics are very close to Logistic Regression


# Model Comparison

# Accuracy Score for each Model
model = [LogReg, DT, RF, Bag, KNN_best, svc_best]
comp = []
for i in model:
    print( i,'=', accuracy_score(target_test, i.predict(pca_test_df)))

# Confusion Matrix for each model
from sklearn import metrics
for i in model:
    print(i)
    cm_i = confusion_matrix(target_test, i.predict(pca_test_df))
    cm_display_i = metrics.ConfusionMatrixDisplay(confusion_matrix = cm_i, display_labels = ["Benign", "Malignant"])
    cm_display_i.plot()
    plt.show()

# Random Forest with PCA has the least number of False Negative and False Positive values


# Classification Report of each model
for i in model:
    print(i)
    print(classification_report(target_test, i.predict(pca_test_df)))


# Conclusion

# 1. We can see that Random Forest with PCA has the highest accuracy, precision, recall, and f1 score of all the models. Hence we    can go with PCA + RF as our final model.
# 2. The model has a sensitivity of 0.97, Precision of 0.98 and accuracy of 0.98
# 3. We can try stacking different models to check if the metrics can be improved upon.
