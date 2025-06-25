# project/preprocessing.py
# Text preprocessing utilities for the project/ directory
# Uses spaCy for tokenization, stopword removal, punctuation removal, and lemmatization

import spacy
from typing import List

# Load spaCy English model (make sure to install: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000  # or higher if needed


def spacy_preprocess(text: str) -> str:
    """
    Clean and tokenize text using spaCy. Removes stopwords, punctuation, and lemmatizes tokens.
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)


def batch_preprocess(texts: List[str]) -> List[str]:
    """
    Apply spaCy preprocessing to a list of texts.
    """
    return [spacy_preprocess(t) for t in texts]
