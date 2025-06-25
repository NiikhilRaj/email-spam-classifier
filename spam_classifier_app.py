import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model, vectorizer, and transform_text
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()


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


st.title('Email Spam Classifier')
st.write('Type or paste your email message below:')

user_input = st.text_area('Email Text', height=200)

if st.button('Predict'):
    if user_input.strip() == '':
        st.warning('Please enter some text.')
    else:
        # Preprocess and vectorize
        transformed = transform_text(user_input)
        vect = vectorizer.trasform([transformed]).toarray()
        prediction = model.predict(vect)[0]
        if prediction == 1:
            st.error('Spam!')
        else:
            st.success('Not Spam (Ham)')
