#!/bin/bash
# Email Spam Classifier Setup Script
# This script sets up a virtual environment, installs dependencies, downloads NLTK data, and runs the Streamlit app.

set -e

# 1. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies
pip install -r <(echo "streamlit pandas numpy scikit-learn matplotlib seaborn nltk")

# 4. Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Run Streamlit app
echo "Launching Streamlit app..."
streamlit run spam_classifier_app.py
