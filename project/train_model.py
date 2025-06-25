# project/train_model.py
# Model training script for the project/ directory
# Loads data, preprocesses text, splits data, trains models, evaluates, and saves model/vectorizer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os
from preprocessing import batch_preprocess

# Config: set data and output paths
DATA_PATH = os.path.join(os.path.dirname(
    __file__), '..', '190k_mails_spam.csv')  # Change if needed
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

# Load data (expects a CSV with text and label columns)
df = pd.read_csv(DATA_PATH)
# Robust column handling: auto-detect text and label columns
df.columns = [c.lower() for c in df.columns]
text_col = next(
    (c for c in df.columns if 'text' in c or 'message' in c or 'email' in c), df.columns[0])
label_col = next(
    (c for c in df.columns if 'label' in c or 'spam' in c or 'target' in c), df.columns[-1])

# Drop missing values
df = df[[text_col, label_col]].dropna()

# Preprocess text using spaCy utilities
print('Preprocessing text...')
df['clean_text'] = batch_preprocess(df[text_col].astype(str).tolist())

# Split data (66% train, 22% test, 12% val)
train_df, temp_df = train_test_split(
    df, test_size=0.34, random_state=42, stratify=df[label_col])
test_df, val_df = train_test_split(
    # 0.353*0.34 ≈ 0.12
    temp_df, test_size=0.353, random_state=42, stratify=temp_df[label_col])

# Vectorize text (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])
X_val = vectorizer.transform(val_df['clean_text'])
y_train = train_df[label_col]
y_test = test_df[label_col]
y_val = val_df[label_col]

# Train models: MultinomialNB and RandomForest
print('Training MultinomialNB...')
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

print('Training RandomForest...')
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# VotingClassifier combines both models
voting = VotingClassifier(estimators=[('mnb', mnb), ('rf', rf)], voting='soft')
voting.fit(X_train, y_train)

# Evaluate on test and validation sets
print('Test set results:')
y_pred = voting.predict(X_test)
print(classification_report(y_test, y_pred))

print('Validation set results:')
y_val_pred = voting.predict(X_val)
print(classification_report(y_val, y_val_pred))

# Export model and vectorizer for use in CLI and apps
joblib.dump({'model': voting, 'vectorizer': vectorizer}, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f'Model and vectorizer saved to {MODEL_PATH} and {VECTORIZER_PATH}')
