"""
IMDb Sentiment Classifier - Sentiment Predictor

This script loads the trained model and vectorizer to predict sentiment on new movie reviews.
"""

import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK data is available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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

def predict_sentiment(review_text, model, vectorizer, preprocess=True):
    """
    Predict sentiment for a new review
    
    Args:
        review_text (str): Raw text of the review
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        preprocess (bool): Whether to preprocess the text first
    
    Returns:
        sentiment (str): 'Positive' or 'Negative'
        probability (float): Confidence in prediction (0-1)
    """
    # Preprocess if requested
    if preprocess:
        processed_text = preprocess_text(review_text)
    else:
        processed_text = review_text
    
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
    
    return sentiment, probability, processed_text

def main():
    """Main function to run the sentiment predictor"""
    print("IMDb Sentiment Classifier - Sentiment Predictor")
    print("-" * 50)
    
    try:
        # Load the model and vectorizer
        print("Loading model and vectorizer...")
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("Model and vectorizer loaded successfully.")
        
        # Interactive mode
        print("\nEnter a movie review to predict sentiment (type 'exit' to quit):")
        while True:
            user_input = input("\n> ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting the sentiment predictor.")
                break
            
            if not user_input.strip():
                print("Please enter a review or type 'exit' to quit.")
                continue
            
            # Make prediction
            sentiment, confidence, processed_text = predict_sentiment(user_input, model, vectorizer)
            
            # Format the confidence as a percentage
            confidence_percent = confidence * 100
            
            # Print the result with visual indicator
            print(f"\nPrediction: {sentiment}")
            print(f"Confidence: {confidence_percent:.1f}%")
            
            # Add a visual confidence bar
            bar_length = 30
            filled_length = int(bar_length * confidence)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"Confidence bar: |{bar}|")
            
            # Show preprocessed text if requested
            print(f"\nPreprocessed text: {processed_text}")
            
            print("\nEnter another review or type 'exit' to quit:")
    
    except FileNotFoundError:
        print("Error: Model or vectorizer file not found.")
        print("Please run the training script (03_vectorization_and_training.py) first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 