# project/predict.py
# Predict script for the project/ directory
# Loads trained model and vectorizer, preprocesses input text, predicts spam/ham, and prints results

import joblib
import os
from preprocessing import batch_preprocess

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_model.pkl')


def predict(texts):
    """
    Predict spam/ham for a list of texts.
    Loads model and vectorizer, preprocesses input, and returns predictions.
    """
    # Load model and vectorizer
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle['model']
    vectorizer = model_bundle['vectorizer']
    # Preprocess
    clean_texts = batch_preprocess(texts)
    X = vectorizer.transform(clean_texts)
    # Predict
    preds = model.predict(X)
    return preds


if __name__ == "__main__":
    # Example usage: python predict.py 'your email text here'
    import sys
    if len(sys.argv) > 1:
        input_texts = sys.argv[1:]
        results = predict(input_texts)
        for text, label in zip(input_texts, results):
            print(f"Text: {text}\nPrediction: {label}\n")
    else:
        print("Usage: python predict.py 'your email text here'")
