"""
Movie Reviews Sentiment Classifier - Vectorization and Model Training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import time
import os

# Ensure photos directory exists
if not os.path.exists('photos'):
    os.makedirs('photos')

def main():
    """Main function for text vectorization and model training"""
    print("Movie Reviews Sentiment Classifier - Vectorization and Model Training")
    print("-" * 70)
    
    # Load preprocessed datasets
    print("Loading preprocessed datasets...")
    train_df = pd.read_csv('preprocessed_train.csv')
    val_df = pd.read_csv('preprocessed_val.csv')
    test_df = pd.read_csv('preprocessed_test.csv')
    print(f"Loaded {len(train_df)} training examples, {len(val_df)} validation examples, and {len(test_df)} test examples.")
    
    # Prepare data for model training
    X_train = train_df['cleaned_text']
    y_train = train_df['label']
    
    X_val = val_df['cleaned_text']
    y_val = val_df['label']
    
    X_test = test_df['cleaned_text']
    y_test = test_df['label']
    
    # Vectorization with TF-IDF
    print("\nVectorizing text using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit features to top 10,000
        min_df=5,            # Ignore terms that appear in less than 5 documents
        max_df=0.8,          # Ignore terms that appear in more than 80% of documents
        ngram_range=(1, 2)   # Include unigrams and bigrams
    )
    
    # Fit vectorizer on training data only
    start_time = time.time()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform validation and test data using the fitted vectorizer
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    vectorization_time = time.time() - start_time
    print(f"Vectorization complete in {vectorization_time:.2f} seconds.")
    print(f"Number of features (vocabulary size): {len(tfidf_vectorizer.get_feature_names_out())}")
    print(f"Training data shape: {X_train_tfidf.shape}")
    
    # Get top features by TF-IDF score
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("\nTop 20 features by TF-IDF score:")
    # Sum TF-IDF scores across all documents for each feature
    tfidf_sums = X_train_tfidf.sum(axis=0).A1
    top_indices = np.argsort(tfidf_sums)[-20:]
    for idx in reversed(top_indices):
        print(f"{feature_names[idx]}: {tfidf_sums[idx]:.2f}")
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Linear SVM': LinearSVC(random_state=42, max_iter=10000, C=1.0)
    }
    
    # Train and evaluate each model
    print("\nTraining and evaluating models:")
    model_results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # Train the model
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        start_time = time.time()
        y_val_pred = model.predict(X_val_tfidf)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        
        # Store results
        model_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'inference_time': inference_time,
            'model': model
        }
        
        # Print metrics
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Inference time: {inference_time:.4f} seconds for {len(X_val)} samples")
        
        # Print classification report
        print("\n  Classification Report:")
        print(classification_report(y_val, y_val_pred, target_names=['Negative', 'Positive']))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(f'photos/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
    
    # Find the best model based on F1-score
    best_model_name = max(model_results, key=lambda x: model_results[x]['f1_score'])
    best_model = model_results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with F1-Score of {model_results[best_model_name]['f1_score']:.4f}")
    
    # Evaluate the best model on the test set
    print("\nEvaluating the best model on the test set...")
    y_test_pred = best_model.predict(X_test_tfidf)
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Print test metrics
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    
    # Print test classification report
    print("\n  Test Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive']))
    
    # Plot test confusion matrix
    test_cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Test Confusion Matrix - {best_model_name}')
    plt.savefig('photos/test_confusion_matrix.png')
    plt.close()
    
    # Compare model performances
    model_names = list(model_results.keys())
    accuracies = [model_results[model]['accuracy'] for model in model_names]
    f1_scores = [model_results[model]['f1_score'] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score')
    
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('photos/model_comparison.png')
    plt.close()
    
    # Save the best model and vectorizer
    print("\nSaving the best model and vectorizer...")
    joblib.dump(best_model, 'sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved.")
    
    # Create a sample prediction function and test it
    print("\nTesting prediction function with sample reviews...")
    
    # Function to predict sentiment
    def predict_sentiment(review_text, model=best_model, vectorizer=tfidf_vectorizer):
        """
        Predict sentiment for a new review
        
        Args:
            review_text: Raw text of the review
            model: Trained classifier
            vectorizer: Fitted TF-IDF vectorizer
            
        Returns:
            sentiment: 'Positive' or 'Negative'
            probability: Confidence in prediction (0-1)
        """
        # Vectorize the text
        text_tfidf = vectorizer.transform([review_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        
        # Get probability if the model supports it
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(text_tfidf)[0][prediction]
        else:
            # For models like LinearSVC that don't have predict_proba
            decision = model.decision_function(text_tfidf)[0]
            probability = 1 / (1 + np.exp(-abs(decision)))
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return sentiment, probability
    
    # Sample positive review
    positive_review = "This is one of the best movies I've seen. The acting was superb and the plot was captivating. Highly recommended!"
    # Sample negative review
    negative_review = "I hated this movie. The plot made no sense and the acting was terrible. Complete waste of time."
    
    # Make predictions
    pos_sentiment, pos_probability = predict_sentiment(positive_review)
    neg_sentiment, neg_probability = predict_sentiment(negative_review)
    
    # Print results
    print(f"Positive review: '{positive_review[:50]}...'")
    print(f"  Predicted: {pos_sentiment} with confidence {pos_probability:.4f}")
    
    print(f"Negative review: '{negative_review[:50]}...'")
    print(f"  Predicted: {neg_sentiment} with confidence {neg_probability:.4f}")
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main() 