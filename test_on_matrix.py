# test_on_matrix.py
# Script to test a trained model on a matrix-form dataset (e.g., emails.csv)
# Loads model and data, predicts, and prints results/accuracy

import pandas as pd
import pickle

# Load the trained model (expects model.pkl)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the matrix-form dataset (expects emails.csv)
df_new = pd.read_csv('emails.csv')

# If you have a label column, specify it here. Otherwise, use all columns as features.
# For example, if the label column is named 'target', uncomment the next two lines:
# y_true = df_new['target'].values
# X_new = df_new.drop(columns=['target']).values

# If your last column is the label, use this:
if df_new.shape[1] > 1:
    y_true = df_new.iloc[:, -1].values
    X_new = df_new.iloc[:, :-1].values
else:
    y_true = None
    X_new = df_new.values

# Predict using the loaded model
predictions = model.predict(X_new)

print("Predictions:", predictions)

# Print accuracy if labels are present
if y_true is not None:
    from sklearn.metrics import accuracy_score
    print("Accuracy:", accuracy_score(y_true, predictions))
