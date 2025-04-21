"""
IMDb Sentiment Classifier - Data Loading and Exploration
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import nltk

# Set plotting style
plt.style.use('ggplot')

def main():
    """Main function for data loading and exploration"""
    print("IMDb Sentiment Classifier - Data Loading and Exploration")
    print("-" * 50)
    
    # Download necessary NLTK data
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully.")
    
    # Load the IMDb dataset from Hugging Face
    print("\nLoading IMDb dataset from Hugging Face...")
    imdb_dataset = load_dataset("imdb")
    print("Dataset loaded successfully!")
    print(f"Dataset structure: {imdb_dataset}")
    
    # Convert to pandas DataFrames for easier handling
    train_df = pd.DataFrame(imdb_dataset['train'])
    test_df = pd.DataFrame(imdb_dataset['test'])
    
    # Display sample data
    print("\nSample training data:")
    print(train_df.head())
    
    # Display basic dataset information
    print("\nBasic information about the training set:")
    print(f"Number of reviews: {len(train_df)}")
    print(f"Columns: {train_df.columns.tolist()}")
    
    # Check class distribution
    print("\nClass distribution in training set:")
    label_counts = train_df['label'].value_counts()
    print(label_counts)
    print(f"Percentage of positive reviews: {label_counts[1] / len(train_df) * 100:.2f}%")
    print(f"Percentage of negative reviews: {label_counts[0] / len(train_df) * 100:.2f}%")
    
    # Check if class distribution is balanced in test set as well
    test_label_counts = test_df['label'].value_counts()
    print("\nClass distribution in test set:")
    print(test_label_counts)
    
    # Add review length as a feature
    train_df['review_length'] = train_df['text'].apply(len)
    
    # Display basic statistics about review length
    print("\nStatistics about review lengths in training set:")
    print(train_df['review_length'].describe())
    
    # Plot review length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x='review_length', hue='label', bins=50, alpha=0.6)
    plt.title('Distribution of Review Lengths by Sentiment')
    plt.xlabel('Review Length (characters)')
    plt.ylabel('Count')
    plt.savefig('review_length_distribution.png')
    plt.close()
    print("Review length distribution plot saved as 'review_length_distribution.png'")
    
    # Display a sample positive and negative review
    pos_sample = train_df[train_df['label'] == 1].iloc[0]['text']
    neg_sample = train_df[train_df['label'] == 0].iloc[0]['text']
    
    print("\nSample positive review (truncated):")
    print(pos_sample[:500] + "...")
    
    print("\nSample negative review (truncated):")
    print(neg_sample[:500] + "...")
    
    # Save the dataset for further processing
    train_df.to_csv('imdb_train.csv', index=False)
    test_df.to_csv('imdb_test.csv', index=False)
    print("\nDatasets saved as CSV files for further processing.")
    
    print("\nData exploration complete!")
    
if __name__ == "__main__":
    main() 