# Movie Reviews Sentiment Classifier

A sentiment analysis project that classifies movie reviews as positive or negative using NLP techniques.

## Overview

This project builds a machine learning model to classify movie reviews from the IMDb dataset as either positive or negative. The model uses natural language processing (NLP) techniques to understand and classify review sentiment.

## Features

- Data loading and exploration of the IMDb dataset
- Text preprocessing (removing HTML tags, stopwords, etc.)
- Text vectorization using TF-IDF
- Model training and evaluation with multiple classifiers (Logistic Regression, Naive Bayes, SVM)
- Interactive sentiment prediction via:
  - Command-line interface for testing
  - Web application with modern UI

## Project Structure

- **Data Processing & Model Training:**
  - `01_data_loading.py` - Loads and explores the IMDb dataset
  - `02_preprocessing.py` - Preprocesses the text data (cleaning, tokenization, etc.)
  - `03_vectorization_and_training.py` - Vectorizes the text and trains classification models
  - `04_sentiment_predictor.py` - Provides a command-line interface to test the model
  - `download_nltk_data.py` - Downloads required NLTK data packages

- **Web Application:**
  - `app.py` - Main Flask application serving the web interface
  - `templates/index.html` - HTML template for the web app
  - `static/css/style.css` - Custom CSS styles
  - `static/js/script.js` - JavaScript for the web interface

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- joblib
- datasets (HuggingFace)
- Flask (for web application)

## Installation

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Download required NLTK data:
   ```
   python download_nltk_data.py
   ```

## Usage

### 1. Data Loading and Exploration

```
python 01_data_loading.py
```

This script:
- Downloads the IMDb dataset from HuggingFace
- Explores the dataset characteristics
- Saves the data as CSV files for further processing

### 2. Text Preprocessing

```
python 02_preprocessing.py
```

This script:
- Loads the IMDb dataset
- Creates a validation set
- Cleans and preprocesses the text (removing HTML, tokenization, lemmatization, etc.)
- Saves the preprocessed datasets for model training

### 3. Vectorization and Model Training

```
python 03_vectorization_and_training.py
```

This script:
- Vectorizes the text using TF-IDF
- Trains multiple models (Logistic Regression, Naive Bayes, SVM)
- Evaluates and compares model performance
- Saves the best model and vectorizer

### 4. Command-line Sentiment Prediction

```
python 04_sentiment_predictor.py
```

This script:
- Loads the trained model and vectorizer
- Provides an interactive command-line interface to enter movie reviews
- Predicts the sentiment (positive/negative) with confidence scores

### 5. Web Application

```
python app.py
```

This command:
- Starts the Flask web server
- Opens a web interface accessible at http://127.0.0.1:8080
- Allows users to input reviews and see prediction results with a modern UI

## Results

The trained models achieve high accuracy (~85-90%) on the IMDb dataset. Visualizations of model performance and confusion matrices are generated during the training process.

## Web Application Screenshots

The web application provides a user-friendly interface for sentiment analysis:

- **Input Screen:** Clean form for entering movie reviews
- **Results Display:** Shows sentiment prediction, confidence score, and preprocessing details
- **Responsive Design:** Works well on both desktop and mobile devices

## License

This project is available under the MIT License.

## Acknowledgments

- The IMDb dataset is sourced from HuggingFace datasets
- This project was created as part of an NLP course assignment 