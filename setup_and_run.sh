#!/bin/bash
# Email Spam Classifier Setup Script
# This script sets up a virtual environment, installs dependencies, downloads NLTK and spaCy data, and provides commands to run all main scripts and notebooks.

set -e

# 1. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies (add spaCy, joblib, and notebook for full compatibility)
pip install streamlit pandas numpy scikit-learn matplotlib seaborn nltk spacy joblib notebook

# 4. Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Download spaCy model for project/ scripts
echo "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

# 6. Usage instructions
echo """
Setup complete! You can now run:
- Streamlit app:         streamlit run spam_classifier_app.py
- Classic app:           streamlit run app.py
- Export model:          python export_model.py
- Test on matrix:        python test_on_matrix.py
- Project CLI:           cd project && python app.py train|predict 'your text here'
- Project notebooks:     jupyter notebook
"""
