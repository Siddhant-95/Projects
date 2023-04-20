
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')

df = pd.read_csv('Life Expectancy Data.csv')
df

df.dtypes

df.info()

df = df[df['Life expectancy '].notna()]

df[df.duplicated()]

df.info()

plt.figure(figsize = (10,7))
cols = df.columns
sns.heatmap(df[cols].isnull())
sns.displot(df['Life expectancy '], kde = True, color = 'green')

for i in df.columns:
    try: 
        mean = df[i].mean()
        df[i] = df[i].fillna(mean)
    except:
        pass

df.info()


# Country vs Life expectancy

c_vs_life = df.groupby('Country')['Life expectancy '].mean()
c_vs_life.sort_values(ascending= False).head(5)

c_vs_life.sort_values().head(5)

# Status vs Life Expectancy

df.Status.value_counts()

le_means = df.groupby('Status')['Life expectancy '].mean()
le_means

status = ('Developing','Developed')
plt.bar(status, le_means, color = ['red','green'])
plt.show()


# Life Expectancy vs Numerical features

numerics = ['int64', 'float64']
df_num = df.select_dtypes(include = numerics)
df_num

df_num.corr()

plt.figure(figsize = (9,5))
sns.heatmap(df_num.corr(), center = 0)

for i in df_num.columns:
    sns.regplot(x = df[i], y = df['Life expectancy '], color = 'blue')
    plt.show()


# Model Selection

target = df['Life expectancy ']
features = df.loc[:, df.columns != 'Life expectancy ']
features
target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(pd.get_dummies(features), target, test_size = 0.33, random_state = 9)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, max_features = 'auto', random_state = 9)
model = rf.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_predict, Y_test, squared = False)

error = abs(Y_predict - Y_test)
mape = 100 * np.mean(error/Y_test)
accuracy = 100 - mape
accuracy

base_rf = RandomForestRegressor()
model_base = base_rf.fit(X_train, Y_train)

Y_pred = model_base.predict(X_test)

mean_squared_error(Y_pred, Y_test, squared = False)
Y_pred = model_base.predict(X_test)
error_base = abs(Y_pred - Y_test)
mape_base = 100 * np.mean(error_base/Y_test)
accuracy_base = 100 - mape_base
accuracy_base
