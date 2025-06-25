# spam_classifier_app.py
# Streamlit app for classifying emails as spam or not spam
# Loads trained model and vectorizer, preprocesses input, predicts, and displays result

import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model, vectorizer, and transform_text function
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()

# Text preprocessing: lowercasing, tokenization, removing stopwords/punctuation, stemming
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(stemmer.stem(i))
    return " ".join(y)

# Streamlit UI
description = """
This app classifies email text as Spam or Not Spam using a trained machine learning model.
"""
st.title('Email Spam Classifier')
st.write(description)
st.write('Type or paste your email message below:')

user_input = st.text_area('Email Text', height=200)

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter some text.')
    else:
        # Preprocess and vectorize
        transformed = transform_text(user_input)
        vect = vectorizer.transform([transformed]).toarray()
        prediction = model.predict(vect)[0]
        if prediction == 1:
            st.error('Spam!')
        else:
            st.success('Not Spam (Ham)')
