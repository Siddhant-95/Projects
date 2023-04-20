import pandas as pd
import numpy as np
import re
import nltk
from pickle import dump
from pickle import load

from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')
sw_list = ['ye','yeah','haha','Yes','I']
my_stop_words.extend(sw_list)

tweets = pd.read_csv('P96 all tweets.csv')
tweets.drop(['Unnamed: 0','user_location','date','hashtags','keyword'], axis = 1, inplace = True)
tweets = tweets.drop_duplicates(keep = False)
tweets = tweets.reset_index()
tweets.drop(['index'], axis = 1, inplace = True)

tweets_ml = tweets.copy()

tweets_ml.text = tweets_ml.text.apply(lambda x: re.sub(r'http\S+', '',x)) #remove Urls
tweets_ml.text = tweets_ml.text.apply(lambda x: re.sub('@[^\s]+','',x)) #remove handles
tweets_ml.text = tweets_ml.text.apply(lambda x: re.sub('[^a-zA-Z0-9]'," ",x)) #remove everything except letters and numbers
tweets_ml.text = tweets_ml.text.apply(lambda x : x.split())
tweets_ml.text = tweets_ml.text.apply(lambda x: [word for word in x if word not in set(stopwords.words('english'))])

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
tweets_ml.text = tweets_ml.text.apply(lambda x : [ps.stem(word) for word in x]) #stemming
tweets_ml.text = tweets_ml.text.apply(lambda x : " ".join(x))

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer()
X = tv.fit_transform(tweets_ml.text.values)
X = pd.DataFrame(X.todense(),columns = tv.get_feature_names_out())
Y = tweets_ml.LABEL

#save features to disk
dump(tv.vocabulary_, open('features.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 9)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 500, max_features = 20)
model.fit(X_train, Y_train)

#save model to disk
dump(model, open('RF.sav','wb'))

#load the model from disk
loaded_model = load(open('RF.sav','rb'))
result = loaded_model.score(X, Y)
print(result)



