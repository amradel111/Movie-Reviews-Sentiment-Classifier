# IMDb Sentiment Classifier

A sentiment analysis project that classifies movie reviews as positive or negative using NLP techniques.

## Overview

This project builds a machine learning model to classify movie reviews from the IMDb dataset as either positive or negative. The model uses natural language processing (NLP) techniques to understand and classify review sentiment.

## Features

- Data loading and exploration of the IMDb dataset
- Text preprocessing (removing HTML tags, stopwords, etc.)
- Text vectorization using TF-IDF
- Model training and evaluation with multiple classifiers (Logistic Regression, Naive Bayes, SVM)
- Interactive sentiment prediction for new reviews

## Project Structure

- `01_data_loading.py` - Loads and explores the IMDb dataset
- `02_preprocessing.py` - Preprocesses the text data (cleaning, tokenization, etc.)
- `03_vectorization_and_training.py` - Vectorizes the text and trains classification models
- `04_sentiment_predictor.py` - Provides an interactive interface to test the model
- `download_nltk_data.py` - Downloads required NLTK data packages

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
   pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib datasets
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

### 4. Sentiment Prediction

```
python 04_sentiment_predictor.py
```

This script:
- Loads the trained model and vectorizer
- Provides an interactive interface to enter movie reviews
- Predicts the sentiment (positive/negative) with confidence scores

## Results

The trained models achieve high accuracy (~85-90%) on the IMDb dataset. Visualizations of model performance and confusion matrices are generated during the training process.

## License

This project is available under the MIT License.

## Acknowledgments

- The IMDb dataset is sourced from HuggingFace datasets
- This project was created as part of an NLP course assignment 