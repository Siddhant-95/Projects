import pandas as pd
import numpy as np
import re
import nltk
from pickle import dump
from pickle import load
import streamlit as st

from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')
sw_list = ['ye','yeah','haha','Yes','I']
my_stop_words.extend(sw_list)

tweets = pd.read_csv('P96 new disaster.csv') #reading the dataset with extracted tweets
tweets.drop(['Unnamed: 0','user_name','user_description','user_verified','source'], axis = 1, inplace = True)
tweets.drop(['user_location','date','hashtags'], axis = 1, inplace = True)
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
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
loaded_vec = TfidfVectorizer(vocabulary = load(open('features.pkl', 'rb')))
X = transformer.fit_transform(loaded_vec.fit_transform(tweets_ml.text.values))

st.subheader('DataFrame')
st.write(tweets)

loaded_model = load(open('RF.sav','rb'))
prediction = loaded_model.predict(X)
prediction_proba = loaded_model.predict_proba(X)
st.subheader('Predicted Probability')
st.write(prediction_proba)

output = pd.concat([tweets, pd.DataFrame(prediction_proba)], axis = 1)
output.to_csv('output.csv')
