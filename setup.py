"""
Setup script to initialize required NLTK data and verify dependencies.
"""

import nltk
import logging

def setup_nltk():
    """Download required NLTK data."""
    try:
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    setup_nltk()