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

st.subheader('Input Tweet')
tweet = st.text_input('Tweet')

def tweet_to_data(x):
    return pd.DataFrame({'text' : x}, index = [0])

tweets = tweet_to_data(tweet)

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

st.subheader('Tweet')
st.write(tweet)

loaded_model = load(open('RF.sav','rb'))
prediction = loaded_model.predict(X)
prediction_proba = loaded_model.predict_proba(X)
st.subheader('Predicted Probability')

prediction = pd.DataFrame(prediction_proba)
st.write(prediction)

if prediction.iloc[0,1] > 0.5:
    st.write("Disaster Tweet")
else:
    st.write("Not a Disaster Tweet")
    