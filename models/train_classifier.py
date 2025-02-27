import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split

def load_data(database_filepath):
    """Loads data from the SQLite database and splits it into features (X) and labels (Y)."""
    
    # Loading database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessagesCategories', engine)  

    # Defining feature (X) and labels (Y)
    X = df['message']  # Feature: Message text
    Y = df.iloc[:, 4:]  # Labels: All categories after column index 4

    # Extracting category names
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text by normalizing, removing punctuation, tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): The input text message.

    Returns:
        List[str]: A list of processed tokens.
    """

    # Normalize text (convert to lowercase & remove punctuation)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenizing text
    tokens = word_tokenize(text)

    # Removing stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatizing words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens]

    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline with GridSearchCV.

    Returns:
        model: GridSearchCV model pipeline
    """

    # Define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Convert text to word counts
        ('tfidf', TfidfTransformer()),  # Convert counts to TF-IDF scores
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))  # Classifier
    ])

    # Define hyperparameters to tune
    parameters = {
        'clf__estimator__n_estimators': [50, 100],  # Number of trees in RF
        'clf__estimator__min_samples_split': [2, 4]  # Minimum samples per split
    }

    # Initialize GridSearchCV
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=2, n_jobs=-1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance on the test set.
    
    Args:
    model -- Trained ML model
    X_test -- Test features
    Y_test -- True labels
    category_names -- List of category names
    
    Returns:
    None (Prints evaluation metrics)
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Loop through each category and print precision, recall, and f1-score
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        print("-" * 60)  # Separator for readability


def save_model(model, model_filepath):
    """ Save the trained model as a pickle file """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()