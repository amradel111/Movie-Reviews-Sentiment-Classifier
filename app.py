"""
IMDb Sentiment Classifier - Web Application
"""

import os
import re
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging
import sys

# --- Setup Logging --- 
try:
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler = logging.FileHandler('app_debug.log', mode='w') # Overwrite log each time
    log_handler.setFormatter(log_formatter)
    
    # Log to file
    file_logger = logging.getLogger('file_logger')
    file_logger.setLevel(logging.DEBUG)
    file_logger.addHandler(log_handler)
    
    # Also log basic info to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_logger = logging.getLogger('console_logger')
    console_logger.setLevel(logging.INFO) 
    console_logger.addHandler(console_handler)
    
    file_logger.info("--- Script app.py started ---")
    console_logger.info("--- Script app.py started ---")
    
except Exception as e:
    print(f"FATAL: Failed to set up logging: {e}")
    # Continue without logging if setup fails
# --- End Logging Setup ---

# Ensure NLTK data is available
try:
    file_logger.info("Checking NLTK data...")
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    file_logger.info("NLTK data found.")
except LookupError:
    file_logger.warning("NLTK data not found, attempting download...")
    console_logger.warning("NLTK data not found, attempting download...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        file_logger.info("NLTK data downloaded.")
    except Exception as e:
        file_logger.error(f"Failed to download NLTK data: {e}")
        console_logger.error(f"Failed to download NLTK data: {e}")

app = Flask(__name__)
app.logger.addHandler(log_handler) # Add file logging to Flask's logger
app.logger.setLevel(logging.INFO)

# Load the model and vectorizer
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

def load_models():
    """Load the trained model and vectorizer"""
    file_logger.info(f"Attempting to load model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    file_logger.info("Model loaded successfully.")
    file_logger.info(f"Attempting to load vectorizer from {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    file_logger.info("Vectorizer loaded successfully.")
    return model, vectorizer

try:
    file_logger.info("Loading models...")
    model, vectorizer = load_models()
    console_logger.info("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    file_logger.error(f"Model/Vectorizer File Not Found: {e}")
    console_logger.error(f"ERROR: Model or vectorizer file not found. Cannot start server.")
    # Exit if models can't be loaded
    sys.exit(f"ERROR: {e}")
except Exception as e:
    file_logger.error(f"Unexpected error loading models: {e}", exc_info=True)
    console_logger.error(f"ERROR: An unexpected error occurred while loading models: {e}")
    sys.exit(f"ERROR loading models: {e}")

def simple_tokenize(text):
    """Simple tokenization using regex"""
    # Convert to lowercase and replace punctuation and numbers with spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Split on whitespace and filter out empty tokens
    tokens = [token.strip() for token in text.split()]
    return [token for token in tokens if token]

def preprocess_text(text):
    """
    Preprocess the text for the model
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Tokenize using simple method
    tokens = simple_tokenize(text)
    
    # Get English stopwords but keep negation words
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'never', 'none', 'nobody', 'nowhere', 'nothing', 'neither'}
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]
    
    # Join tokens back into a string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

def predict_sentiment(review_text):
    """
    Predict sentiment for a new review
    
    Args:
        review_text (str): Raw text of the review
        
    Returns:
        dict: Dictionary with sentiment and confidence
    """
    try:
        # Preprocess the text
        processed_text = preprocess_text(review_text)
        
        if not processed_text:
            file_logger.warning(f"Preprocessing resulted in empty text for input: {review_text[:50]}...")
            return {
                "sentiment": "Unknown",
                "confidence": 0,
                "processed_text": "Text too short or contains only stopwords after preprocessing.",
                "error": "Please provide more meaningful text."
            }
            
        # Vectorize the text
        text_tfidf = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        
        # Get probability/confidence if the model supports it
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(text_tfidf)[0][prediction]
        else:
            # For models like LinearSVC that don't have predict_proba
            decision = model.decision_function(text_tfidf)[0]
            probability = 1 / (1 + np.exp(-abs(decision)))
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        file_logger.info(f"Prediction for '{review_text[:50]}...': {sentiment} ({probability:.4f})")
        return {
            "sentiment": sentiment,
            "confidence": float(probability),
            "processed_text": processed_text
        }
    except Exception as e:
        file_logger.error(f"Error during prediction: {e}", exc_info=True)
        return {
            "sentiment": "Error",
            "confidence": 0,
            "error": str(e)
        }

@app.route('/')
def index():
    """Render the main page"""
    file_logger.info(f"Request received for route: /")
    models_loaded = True
    # Check again in case something went wrong after initial load
    try:
        if 'model' not in globals() or 'vectorizer' not in globals():
             raise RuntimeError("Models not loaded")
    except Exception:
        models_loaded = False
        file_logger.warning("Models appear not loaded when rendering index.")
    
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction"""
    file_logger.info(f"Request received for route: /predict")
    if not request.is_json:
        file_logger.error("Predict request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'review' not in data:
        file_logger.error("Predict request missing 'review' field")
        return jsonify({"error": "Missing 'review' field"}), 400
    
    review_text = data['review']
    
    if not review_text or not review_text.strip():
        file_logger.warning("Predict request received empty review")
        return jsonify({
            "sentiment": "Unknown",
            "confidence": 0,
            "error": "Please provide a review text."
        }), 400
        
    try:
        result = predict_sentiment(review_text)
        return jsonify(result)
    except Exception as e:
        file_logger.error(f"Unexpected error during /predict processing: {e}", exc_info=True)
        return jsonify({
            "sentiment": "Error",
            "confidence": 0,
            "error": "An internal server error occurred."
        }), 500

if __name__ == '__main__':
    file_logger.info("--- Starting Flask Server --- ")
    # Check if model files exist before trying to load
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        console_logger.error(f"ERROR: Model files ({MODEL_PATH} or {VECTORIZER_PATH}) not found.")
        file_logger.error(f"Model files not found. Aborting server start.")
        sys.exit("Model files not found. Cannot start server.")
    
    try:
        # Start the app
        console_logger.info("=" * 60)
        console_logger.info("Attempting to start IMDb Sentiment Analyzer on http://127.0.0.1:8080")
        console_logger.info("Press Ctrl+C to stop the server")
        console_logger.info("=" * 60)
        app.run(debug=False, host='0.0.0.0', port=8080) # Turn debug off for cleaner logs
    except SystemExit as e:
        console_logger.info(f"Server startup aborted: {e}")
    except Exception as e:
        console_logger.error(f"FATAL: Failed to run Flask app: {e}")
        file_logger.error(f"FATAL: Failed to run Flask app: {e}", exc_info=True)
    finally:
        file_logger.info("--- Flask server process ended ---")
        console_logger.info("--- Server stopped ---") 