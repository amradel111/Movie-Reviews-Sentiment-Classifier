"""
Download all required NLTK data for the IMDb sentiment classifier
"""

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK data...")

# Download all required data packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual WordNet data
nltk.download('punkt_tab')  # Additional punkt data

print("NLTK data download complete.") 