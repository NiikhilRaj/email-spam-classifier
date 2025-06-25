# export_model.py
# Script to train a spam classifier and export model artifacts (vectorizer, model, preprocessing function)
# Loads data, preprocesses text, vectorizes, trains MultinomialNB, and saves artifacts for use in apps.

import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Text preprocessing: lowercasing, tokenization, removing stopwords/punctuation, stemming
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

# Load and clean data (expects 'spam.csv' with columns v1=label, v2=text)
df = pd.read_csv('spam.csv', encoding='latin1', on_bad_lines='skip')
df = df.drop(columns=['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'])
df = df.rename(columns={'v1':'target', 'v2':'text'})
df = df.drop_duplicates(keep='first')

# Encode target labels: ham=0, spam=1
df['target'] = df['target'].map({'ham': 0, 'spam': 1})

# Transform text for ML
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorization (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Export artifacts for use in apps (vectorizer, model, preprocessing)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('transform_text.pkl', 'wb') as f:
    pickle.dump(transform_text, f)

print('Artifacts exported: vectorizer.pkl, model.pkl, transform_text.pkl')
