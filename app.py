# app.py
# Streamlit web app for Email/SMS Spam Classification
# Loads a trained model and vectorizer, preprocesses user input, predicts spam/ham, and displays the result.

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
# vectorizer.pkl: TF-IDF vectorizer
# model.pkl: Trained spam classifier (e.g., MultinomialNB)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit UI
description = """
This app classifies input text (email or SMS) as Spam or Not Spam using a trained machine learning model.
"""
st.title("Email/SMS Spam Classifier")
st.write(description)

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")