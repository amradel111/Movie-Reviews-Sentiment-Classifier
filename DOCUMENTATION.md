# IMDb Sentiment Classifier - Project Documentation

This document provides an overview of the Python scripts used in the IMDb Sentiment Classifier project. Each script represents a distinct step in the process, from data loading to building a predictive web application.

## Project Structure

The project follows a sequential workflow, with each numbered script building upon the output of the previous one:

1.  `01_data_loading.py`: Loads and explores the raw IMDb dataset.
2.  `02_preprocessing.py`: Cleans the text data and prepares train/validation/test splits.
3.  `03_vectorization_and_training.py`: Converts text to numerical features (TF-IDF) and trains/evaluates machine learning models.
4.  `04_sentiment_predictor.py`: Provides a command-line tool for interactive predictions using the trained model.
5.  `app.py`: Implements a Flask web application for a user-friendly prediction interface.

---

## File: `01_data_loading.py`

*   **Purpose**: This script is responsible for acquiring the IMDb movie review dataset, performing initial exploratory data analysis (EDA), and saving the raw data into CSV format for subsequent processing stages.
*   **Key Libraries Used**:
    *   `datasets` (Hugging Face): To easily download and load the standard IMDb dataset.
    *   `pandas`: For data manipulation using DataFrames.
    *   `numpy`: For numerical operations (though less prominent in this script).
    *   `matplotlib` & `seaborn`: For creating visualizations (specifically, the review length distribution).
    *   `nltk`: To download required resources (`punkt`, `stopwords`, `wordnet`) used later or by libraries implicitly.
*   **Core Functionality**:
    1.  Initializes by downloading necessary NLTK data components.
    2.  Uses the `load_dataset("imdb")` function from the `datasets` library to fetch the data.
    3.  Converts the 'train' and 'test' splits of the dataset into pandas DataFrames.
    4.  Prints basic information: sample rows (`head()`), number of reviews, column names.
    5.  Analyzes class distribution (count and percentage of positive/negative reviews) for both training and test sets to check for balance.
    6.  Computes the length (number of characters) of each review and adds it as a new 'review_length' column to the training DataFrame.
    7.  Calculates and prints descriptive statistics (mean, std, min, max, quartiles) for the 'review_length'.
    8.  Generates a histogram showing the distribution of review lengths, colored by sentiment label (positive/negative), and saves it to `photos/review_length_distribution.png`.
    9.  Prints truncated examples of one positive and one negative review for qualitative inspection.
    10. Saves the initial training and test DataFrames into `imdb_train.csv` and `imdb_test.csv`.
*   **Input**: None (downloads the dataset directly from Hugging Face).
*   **Output**:
    *   `imdb_train.csv`: Raw training data with 'text' and 'label' columns.
    *   `imdb_test.csv`: Raw test data with 'text' and 'label' columns.
    *   `photos/review_length_distribution.png`: Histogram visualization.
    *   Console output detailing dataset properties, class distribution, and sample reviews.
*   **Notable Features**: Integration with Hugging Face `datasets` library simplifies data acquisition. Basic but essential EDA steps like checking class balance and analyzing text length are performed. Visualizations aid in understanding data characteristics.

---

## File: `02_preprocessing.py`

*   **Purpose**: This script focuses on cleaning and preparing the raw text data loaded from the CSV files. It performs various text preprocessing steps, splits the training data into training and validation sets, and saves the results.
*   **Key Libraries Used**:
    *   `pandas`: For loading and manipulating the data in DataFrames.
    *   `re`: Python's regular expression module, used heavily for cleaning text (removing HTML, punctuation, numbers).
    *   `nltk`: Specifically `stopwords` (for filtering common words) and `WordNetLemmatizer` (for reducing words to their base form).
    *   `sklearn.model_selection`: Provides the `train_test_split` function for creating validation set.
    *   `matplotlib` & `seaborn`: For visualizing token count distributions.
*   **Core Functionality**:
    1.  Defines a `simple_tokenize` function that converts text to lowercase, removes punctuation and numbers (replacing them with spaces), splits the text by whitespace, and filters out empty tokens.
    2.  Defines the main `preprocess_text` function which orchestrates the cleaning:
        *   Removes HTML tags using `re.sub(r'<.*?>', ' ', text)`.
        *   Applies `simple_tokenize`.
        *   Retrieves the standard English stopwords list from `nltk` but explicitly *removes* negation words (`no`, `not`, `nor`, etc.) from this list to preserve sentiment cues.
        *   Filters out the identified stopwords and tokens with length <= 1.
        *   Applies lemmatization to the remaining tokens using `WordNetLemmatizer`.
        *   Joins the cleaned and lemmatized tokens back into a single string separated by spaces.
    3.  Loads `imdb_train.csv` and `imdb_test.csv`.
    4.  Splits the original training data (`train_df`) into a new, smaller training set (85%) and a validation set (15%) using `train_test_split`. Crucially, it uses `stratify=train_df['label']` to ensure the proportion of positive/negative reviews is maintained in both splits.
    5.  Applies the `preprocess_text` function to the `text` column of the training, validation, and test DataFrames. The results are stored in a new `cleaned_text` column. Processing is done in batches with progress updates for large datasets.
    6.  Prints examples comparing original and preprocessed text for a positive and a negative review.
    7.  Calculates token counts (using `simple_tokenize`) for both original and cleaned text, storing them in new columns. Computes the percentage reduction in tokens.
    8.  Prints average token counts and the average reduction percentage.
    9.  Generates and saves side-by-side histograms (`photos/token_count_distribution.png`) comparing the distribution of token counts before and after preprocessing.
    10. Saves the processed DataFrames (now including `cleaned_text` and token count columns) to `preprocessed_train.csv`, `preprocessed_val.csv`, and `preprocessed_test.csv`.
*   **Input**: `imdb_train.csv`, `imdb_test.csv`.
*   **Output**:
    *   `preprocessed_train.csv`: Training data with 'cleaned_text'.
    *   `preprocessed_val.csv`: Validation data with 'cleaned_text'.
    *   `preprocessed_test.csv`: Test data with 'cleaned_text'.
    *   `photos/token_count_distribution.png`: Token count comparison plot.
    *   Console output with progress, examples, and token statistics.
*   **Notable Features**: Implements a detailed text preprocessing pipeline tailored for sentiment analysis (e.g., keeping negations). Creates a stratified validation set essential for reliable model evaluation. Provides quantitative analysis of the impact of preprocessing on text length.

---

## File: `03_vectorization_and_training.py`

*   **Purpose**: This script takes the preprocessed text data, converts it into numerical feature vectors using TF-IDF, trains several common machine learning classification models, evaluates their performance, selects the best one based on validation results, performs final evaluation on the test set, and saves the best model and the vectorizer for future use.
*   **Key Libraries Used**:
    *   `pandas`: To load the preprocessed data.
    *   `numpy`: For numerical operations, especially with feature matrices and metrics.
    *   `sklearn.feature_extraction.text`: Provides `TfidfVectorizer` for text vectorization.
    *   `sklearn.linear_model`: `LogisticRegression`.
    *   `sklearn.naive_bayes`: `MultinomialNB`.
    *   `sklearn.svm`: `LinearSVC`.
    *   `sklearn.metrics`: Various functions for model evaluation (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`).
    *   `joblib`: For efficiently saving and loading Python objects (the trained model and vectorizer).
    *   `matplotlib` & `seaborn`: For plotting confusion matrices and model comparison charts.
    *   `time`: To measure training and inference times.
*   **Core Functionality**:
    1.  Loads the `preprocessed_train.csv`, `preprocessed_val.csv`, and `preprocessed_test.csv` files.
    2.  Separates the input features (`cleaned_text`) and target labels (`label`) for each dataset split (train, validation, test).
    3.  Initializes `TfidfVectorizer` with specific hyperparameters:
        *   `max_features=10000`: Limits the vocabulary to the 10,000 most frequent terms (based on TF-IDF score).
        *   `min_df=5`: Ignores terms that appear in fewer than 5 reviews.
        *   `max_df=0.8`: Ignores terms that appear in more than 80% of reviews (likely common words or corpus-specific noise).
        *   `ngram_range=(1, 2)`: Includes both single words (unigrams) and adjacent pairs of words (bigrams) as features, capturing some local context.
    4.  Fits the `TfidfVectorizer` *only* on the training data (`X_train`) to learn the vocabulary and IDF weights. Then, transforms the training, validation, and test text data into TF-IDF sparse matrices (`X_train_tfidf`, `X_val_tfidf`, `X_test_tfidf`).
    5.  Prints statistics about the vectorization process (time taken, number of features/vocabulary size, shape of the training matrix).
    6.  Identifies and prints the top 20 features (unigrams or bigrams) based on their cumulative TF-IDF scores across the training corpus.
    7.  Defines a dictionary containing instances of the models to be evaluated: Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machine (LinearSVC). Specific hyperparameters (like `max_iter`, `C`, `alpha`) are set.
    8.  Iterates through each model in the dictionary:
        *   Trains the model using the training TF-IDF matrix and labels (`X_train_tfidf`, `y_train`), measuring the time taken.
        *   Makes predictions on the validation set (`X_val_tfidf`), measuring the inference time.
        *   Calculates standard classification metrics (accuracy, precision, recall, F1-score) by comparing validation predictions (`y_val_pred`) to the true validation labels (`y_val`).
        *   Stores the results (metrics, times, trained model object) in the `model_results` dictionary.
        *   Prints the calculated metrics and training/inference times.
        *   Prints a detailed `classification_report` for the validation set.
        *   Generates and saves a confusion matrix plot for the validation predictions (`photos/confusion_matrix_MODELNAME.png`).
    9.  Determines the best model by finding the one with the highest F1-score on the validation set.
    10. Takes the selected best model and evaluates it on the *unseen* test set (`X_test_tfidf`, `y_test`).
    11. Prints the performance metrics (accuracy, precision, recall, F1) and the classification report for the test set.
    12. Generates and saves the confusion matrix plot for the test set results (`photos/test_confusion_matrix.png`).
    13. Creates and saves a bar chart (`photos/model_comparison.png`) visually comparing the validation accuracy and F1-scores of all trained models.
    14. Uses `joblib.dump` to serialize and save the best performing model object to `sentiment_model.pkl` and the fitted `TfidfVectorizer` object to `tfidf_vectorizer.pkl`.
    15. Includes a helper function `predict_sentiment` demonstrating how to load the saved model/vectorizer and predict sentiment for new, raw text input (including preprocessing within the function). Tests this function with sample positive and negative reviews.
*   **Input**: `preprocessed_train.csv`, `preprocessed_val.csv`, `preprocessed_test.csv`.
*   **Output**:
    *   `sentiment_model.pkl`: The trained machine learning model object (the best one found).
    *   `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer object.
    *   Multiple PNG files in `photos/`: Confusion matrices for validation and test sets, model performance comparison chart.
    *   Console output: Detailed logs of vectorization, top features, model training progress, evaluation metrics for validation and test sets, and sample predictions.
*   **Notable Features**: Implements TF-IDF vectorization including bigrams. Performs systematic comparison of multiple standard classification algorithms (Logistic Regression, Naive Bayes, SVM). Follows best practices by using separate validation (for model selection/tuning) and test (for final unbiased evaluation) sets. Persists the essential artifacts (model and vectorizer) for deployment or later use. Includes confidence score calculation in the sample prediction function.

---

## File: `04_sentiment_predictor.py`

*   **Purpose**: This script serves as a simple, interactive command-line tool that allows a user to get sentiment predictions for movie reviews they type in, using the model and vectorizer saved by the previous script.
*   **Key Libraries Used**:
    *   `joblib`: To load the saved model and vectorizer.
    *   `re`, `nltk` (`stopwords`, `WordNetLemmatizer`): To perform the *exact same* text preprocessing steps as used during training.
    *   `numpy`: Used implicitly for array operations and potentially in the confidence calculation (`np.exp`).
*   **Core Functionality**:
    1.  Re-defines the `simple_tokenize` and `preprocess_text` functions, ensuring consistency with the preprocessing applied before training. (Note: In larger projects, this might be refactored into a shared utility module).
    2.  Defines a `predict_sentiment` function that encapsulates the prediction logic:
        *   Accepts raw review text, the loaded model, and the loaded vectorizer.
        *   Calls `preprocess_text` to clean the input review.
        *   Uses the `vectorizer.transform()` method to convert the cleaned text into a TF-IDF vector.
        *   Uses `model.predict()` to get the sentiment class (0 or 1).
        *   Calculates a confidence score:
            *   If the model has a `predict_proba` method (like Logistic Regression, Naive Bayes), it uses that to get the probability of the predicted class.
            *   If not (like `LinearSVC`), it uses the `decision_function` value. The absolute value of the decision function score relates to the distance from the hyperplane; it's converted to a pseudo-probability between 0.5 and 1 using the formula `1 / (1 + np.exp(-abs(decision)))`.
        *   Translates the predicted class (0/1) into "Negative"/"Positive".
        *   Returns the sentiment string, confidence score, and the processed text.
    3.  The `main` function orchestrates the interactive session:
        *   Attempts to load `sentiment_model.pkl` and `tfidf_vectorizer.pkl` using `joblib.load`. Handles `FileNotFoundError` if they don't exist, instructing the user to run the training script.
        *   Enters a `while True` loop to continuously prompt the user for input.
        *   Reads user input using `input("> ")`.
        *   Checks if the input is an exit command ('exit', 'quit', 'q'). If so, breaks the loop.
        *   If the input is not empty, calls the `predict_sentiment` function.
        *   Prints the predicted sentiment.
        *   Formats the confidence score as a percentage and prints it.
        *   Prints a simple text-based progress bar (`█` and `░`) to visually represent the confidence level.
        *   Prompts for the next review.
*   **Input**:
    *   `sentiment_model.pkl`: Saved trained model.
    *   `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
    *   User input from the command line.
*   **Output**: For each user-provided review, prints the predicted sentiment ("Positive" or "Negative") and a confidence score (percentage and visual bar) to the console.
*   **Notable Features**: Provides a practical way to interact with the trained model. Replicates the exact preprocessing pipeline used during training, which is crucial for correct predictions. Demonstrates how to load and use the saved `joblib` artifacts. Includes a method for estimating confidence even for models like SVM that don't directly output probabilities. Basic user-friendly interaction loop.

---

## File: `app.py`

*   **Purpose**: This script creates a web application using the Flask framework. It exposes the sentiment prediction functionality through a web browser interface, allowing users to submit reviews via a web form and see the predicted sentiment.
*   **Key Libraries Used**:
    *   `Flask`: The micro web framework used to build the application, handle requests, and render templates.
    *   `joblib`: To load the saved model and vectorizer.
    *   `re`, `nltk` (`stopwords`, `WordNetLemmatizer`): For text preprocessing, identical to `04_sentiment_predictor.py`.
    *   `numpy`: For confidence calculation (if using `decision_function`).
    *   `logging`: For detailed logging of application events, requests, and errors to a file (`app_debug.log`) and console.
    *   `sys`: Used for exiting the script if models fail to load and for directing console logs.
*   **Core Functionality**:
    1.  **Logging Setup**: Configures two loggers: one writing detailed `DEBUG` level logs to `app_debug.log` (overwritten on each start), and another writing `INFO` level logs to the console for basic status updates.
    2.  **NLTK Data Check**: Verifies if necessary NLTK data (`stopwords`, `wordnet`) exists locally, attempts to download them if missing, and logs the outcome.
    3.  **Flask App Initialization**: Creates an instance of the Flask application (`app = Flask(__name__)`) and configures its logger.
    4.  **Model Loading**: Defines `load_models` and calls it at startup to load `sentiment_model.pkl` and `tfidf_vectorizer.pkl`. Logs success or failure. If loading fails (e.g., `FileNotFoundError`), it logs the error and exits the script (`sys.exit`) to prevent the server from starting without a functional model.
    5.  **Preprocessing Functions**: Includes the same `simple_tokenize` and `preprocess_text` functions.
    6.  **`predict_sentiment` Function (Web Adapted)**:
        *   Takes the raw review text as input.
        *   Preprocesses the text using `preprocess_text`.
        *   Includes a check: if preprocessing results in an empty string (e.g., input was only stopwords), it returns a specific "Unknown" sentiment and an error message.
        *   Vectorizes the processed text using the loaded `vectorizer`.
        *   Predicts the sentiment using the loaded `model`.
        *   Calculates confidence (handles both `predict_proba` and `decision_function` cases).
        *   Logs the prediction outcome.
        *   Returns a dictionary containing `sentiment`, `confidence` (as float), and `processed_text`. Includes a `try...except` block to catch potential errors during prediction and return an error structure.
    7.  **Web Routes (Endpoints)**:
        *   `@app.route('/')`: Handles requests to the root URL. It renders an HTML template named `index.html` (assumed to be in a `templates` folder). It also passes a flag `models_loaded` to the template.
        *   `@app.route('/predict', methods=['POST'])`: Handles POST requests to the `/predict` endpoint, designed to be called asynchronously (e.g., via JavaScript fetch) from the web page.
            *   Validates that the request is JSON and contains a 'review' field with non-empty text. Returns JSON errors with appropriate HTTP status codes (400) if validation fails.
            *   Extracts the review text.
            *   Calls the web-adapted `predict_sentiment` function.
            *   Returns the result dictionary as a JSON response. Catches unexpected errors during processing and returns a generic server error (500).
    8.  **Main Execution Block (`if __name__ == '__main__':`)**:
        *   Performs a final check for the existence of model files before attempting to start.
        *   Starts the Flask development server using `app.run()`. It's configured to listen on all network interfaces (`host='0.0.0.0'`) on port `8080`, with `debug=False` (recommended for stability, relies on separate logging).
        *   Includes logging for server start/stop events and catches potential exceptions during startup.
*   **Input**:
    *   `sentiment_model.pkl`: Saved trained model.
    *   `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
    *   HTTP GET requests to `/`.
    *   HTTP POST requests (JSON) to `/predict` containing a `{"review": "text..."}` payload.
    *   Requires an HTML file at `templates/index.html` (not shown, but necessary for the `/` route).
*   **Output**:
    *   Serves the HTML interface on `http://<server_ip>:8080/`.
    *   Responds to `/predict` POST requests with JSON data like `{"sentiment": "Positive", "confidence": 0.95, "processed_text": "..."}` or an error structure.
    *   Writes detailed logs to `app_debug.log`.
    *   Prints informational logs to the console.
*   **Notable Features**: Deploys the sentiment analysis model as a web service using Flask. Implements robust logging for monitoring and debugging. Handles model loading securely at startup. Provides a clear API endpoint (`/predict`) for frontend interaction. Includes input validation and error handling for web requests. Reuses the core preprocessing and prediction logic consistently. 