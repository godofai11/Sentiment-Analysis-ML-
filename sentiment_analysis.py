import numpy as np
import pandas as pd
import nltk
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import streamlit as st

nltk.download('stopwords')
from nltk.corpus import stopwords

def openFile(path):
    with open(path) as file:
        data = file.read()
    return data
imdb_data = openFile('imdb_reviews.txt')
amzn_data = openFile('amazon_reviews.txt')
yelp_data = openFile('yelp_reviews.txt')


datasets = [imdb_data, amzn_data, yelp_data]

combined_dataset = []
# Separate samples from each other
for dataset in datasets:
    combined_dataset.extend(dataset.split('\n'))

# Separate each label from each sample
dataset = [sample.split('\t') for sample in combined_dataset]

df = pd.DataFrame(data=dataset, columns=['Reviews', 'Labels'])

# Remove any blank reviews
df = df[df["Labels"].notnull()]

df = df.sample(frac=1)

df['Word Count'] = [len(review.split()) for review in df['Reviews']]

df['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                              for review in df['Reviews']]                           

df['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                            for review in df['Reviews']] 


def getMostCommonWords(reviews, n_most_common, stopwords=None):
    # flatten review column into a list of words, and set each to lowercase
    flattened_reviews = [word for review in reviews for word in \
                         review.lower().split()]

    # remove punctuation from reviews
    flattened_reviews = [''.join(char for char in review if \
                                 char not in string.punctuation) for \
                         review in flattened_reviews]

    # remove stopwords, if applicable
    if stopwords:
        flattened_reviews = [word for word in flattened_reviews if \
                             word not in stopwords]

    # remove any empty strings that were created by this process
    flattened_reviews = [review for review in flattened_reviews if review]

    return Counter(flattened_reviews).most_common(n_most_common)

vectorizer = TfidfVectorizer()
bow = vectorizer.fit_transform(df['Reviews'])
labels = df['Labels']

vectorizer = TfidfVectorizer(min_df=15)
bow = vectorizer.fit_transform(df['Reviews'])
X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.33)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)
classifier.fit(bow,labels) 

sentence = vectorizer.transform(['Worst Product'])

pred=classifier.predict_proba(sentence)
print(pred)


def analyse(text):
    text = vectorizer.transform([text])
    analysis=classifier.predict_proba(text)
    analysis=analysis.item(1)
    if analysis>=0.5:
        st.title("Positive")
    else:
        st.title("Negative")

def sentiment_analysis_webapp():
    st.title("Sentiment Analysis")

    text=st.text_area("Enter the Text :")
    if st.button("Analyse"):
        analyse(text)

sentiment_analysis_webapp()
