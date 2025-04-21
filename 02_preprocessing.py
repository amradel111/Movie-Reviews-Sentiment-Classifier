"""
IMDb Sentiment Classifier - Text Preprocessing
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import ssl

# Force nltk to use unverified SSL context if needed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Make sure necessary NLTK data is downloaded
print("Downloading NLTK data (if needed)...")
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK data check complete.")

def simple_tokenize(text):
    """Simple tokenization using regex instead of NLTK tokenizers"""
    # Convert to lowercase and replace punctuation and numbers with spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Split on whitespace and filter out empty tokens
    tokens = [token.strip() for token in text.split()]
    return [token for token in tokens if token]

def preprocess_text(text):
    """
    Preprocess the text by:
    1. Removing HTML tags
    2. Converting to lowercase
    3. Removing punctuation and numbers
    4. Tokenizing with simple method
    5. Removing stopwords (except negation words)
    6. Lemmatizing
    7. Joining tokens back into a string
    
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

def main():
    """Main function for text preprocessing"""
    print("IMDb Sentiment Classifier - Text Preprocessing")
    print("-" * 50)
    
    # Load the dataset saved from the previous step
    print("Loading the IMDb dataset...")
    train_df = pd.read_csv('imdb_train.csv')
    test_df = pd.read_csv('imdb_test.csv')
    print(f"Loaded {len(train_df)} training examples and {len(test_df)} test examples.")
    
    # Create a validation set from the training data
    print("\nSplitting training data to create a validation set...")
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.15,
        random_state=42,
        stratify=train_df['label']
    )
    print(f"Dataset sizes: Training={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    
    # Apply preprocessing to each dataset
    print("\nApplying text preprocessing...")
    
    # Process a small batch first to check functionality
    print("Processing a small sample as a test...")
    sample_size = 5
    sample_texts = train_df['text'].head(sample_size).apply(preprocess_text)
    print(f"Successfully processed {sample_size} sample texts.")
    
    # Apply preprocessing to training data
    print("Preprocessing training data...")
    # Process in smaller batches to show progress
    batch_size = 1000
    total_batches = len(train_df) // batch_size + (1 if len(train_df) % batch_size > 0 else 0)
    
    cleaned_texts = []
    for i in range(0, len(train_df), batch_size):
        batch = train_df['text'].iloc[i:i+batch_size]
        cleaned_batch = batch.apply(preprocess_text)
        cleaned_texts.extend(cleaned_batch)
        print(f"Processed batch {i//batch_size + 1}/{total_batches}")
    
    train_df['cleaned_text'] = cleaned_texts
    
    # Apply preprocessing to validation data
    print("Preprocessing validation data...")
    val_df['cleaned_text'] = val_df['text'].apply(preprocess_text)
    
    # Apply preprocessing to test data
    print("Preprocessing test data...")
    # Process test data in batches too
    test_batch_size = 1000
    test_total_batches = len(test_df) // test_batch_size + (1 if len(test_df) % test_batch_size > 0 else 0)
    
    test_cleaned_texts = []
    for i in range(0, len(test_df), test_batch_size):
        batch = test_df['text'].iloc[i:i+test_batch_size]
        cleaned_batch = batch.apply(preprocess_text)
        test_cleaned_texts.extend(cleaned_batch)
        print(f"Processed test batch {i//test_batch_size + 1}/{test_total_batches}")
    
    test_df['cleaned_text'] = test_cleaned_texts
    
    print("Preprocessing complete!")
    
    # Display examples of original vs preprocessed text
    print("\nOriginal vs Preprocessed Text Examples:")
    
    # Example from positive review
    pos_idx = train_df[train_df['label'] == 1].index[0]
    print("\nPOSITIVE REVIEW EXAMPLE:")
    print(f"Original: {train_df.loc[pos_idx, 'text'][:200]}...")
    print(f"Preprocessed: {train_df.loc[pos_idx, 'cleaned_text'][:200]}...")
    
    # Example from negative review
    neg_idx = train_df[train_df['label'] == 0].index[0]
    print("\nNEGATIVE REVIEW EXAMPLE:")
    print(f"Original: {train_df.loc[neg_idx, 'text'][:200]}...")
    print(f"Preprocessed: {train_df.loc[neg_idx, 'cleaned_text'][:200]}...")
    
    # Analyze token counts before and after preprocessing
    print("\nCalculating token statistics...")
    # Use our simple tokenizer for consistency
    train_df['original_token_count'] = train_df['text'].apply(lambda x: len(simple_tokenize(x)))
    train_df['cleaned_token_count'] = train_df['cleaned_text'].apply(lambda x: len(simple_tokenize(x)))
    
    # Calculate token reduction percentage
    train_df['token_reduction_pct'] = ((train_df['original_token_count'] - train_df['cleaned_token_count']) / 
                                       train_df['original_token_count'] * 100)
    
    print("\nToken Count Statistics:")
    print(f"Average original token count: {train_df['original_token_count'].mean():.2f}")
    print(f"Average cleaned token count: {train_df['cleaned_token_count'].mean():.2f}")
    print(f"Average token reduction: {train_df['token_reduction_pct'].mean():.2f}%")
    
    # Plot token counts before and after preprocessing
    print("Generating token count distribution plots...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=train_df, x='original_token_count', bins=50, kde=True)
    plt.title('Original Token Count Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x='cleaned_token_count', bins=50, kde=True)
    plt.title('Cleaned Token Count Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('token_count_distribution.png')
    plt.close()
    print("Token count distribution plots saved as 'token_count_distribution.png'")
    
    # Save the preprocessed datasets for model training
    print("\nSaving preprocessed datasets...")
    train_df.to_csv('preprocessed_train.csv', index=False)
    val_df.to_csv('preprocessed_val.csv', index=False)
    test_df.to_csv('preprocessed_test.csv', index=False)
    print("Preprocessed datasets saved as CSV files for model training.")

if __name__ == "__main__":
    main() 