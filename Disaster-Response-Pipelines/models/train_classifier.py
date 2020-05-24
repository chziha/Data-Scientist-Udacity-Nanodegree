import sys
import os
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score

def load_data(database_filepath):
    try:
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        tab_name = database_filepath.split('/')[-1].split('.')[0]
        df = pd.read_sql_table(tab_name, engine)
        X = df['message']
        y = df[df.columns.drop(['id', 'message', 'original', 'genre'])] 
        category_names = y.columns.tolist()
        return X, y, category_names
    except:
         print('Fail to load to the database.')


def tokenize(text):
    '''
    Function to tokenize the text data
    
    Args:
        text: a string
        
    Returns:
        words: a list of tokenized words
    '''

    # Replace all urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)
    for url in urls:
        text.replace(url, 'urlplaceholder')
    
    # Normalize the text
    pattern = r'[^a-zA-Z0-9]'
    text = re.sub(pattern, ' ', text.lower())
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w).strip() for w in tokens]

    # Remove stop words    
    words = [w for w in lemmed if w not in stopwords.words('english')]
    
    return words


def report_metrics(y_true, y_pred, sort_by = None):
    '''
    Function to display the classification report for each category in multi-class classification
    
    Args:
        y_true: a DataFrame with true labels, each row represents the labels for one row in X
        y_pred: 2d array-like with predicted labels, each row represents the labels for one row in X
        sort_by: a string, denoting the column to sort by, select from ['Accuracy', 'F1_score', 'Precision', 'Recall']
        
    Returns:
        df: a Dataframe containing the scores for each category
    '''
    cols = y_true.columns.tolist()
    
    df_list = []
    
    for i, col in enumerate(cols):
        # Replace y_true.to_numpy() with np.array(y_true)
        accuracy = accuracy_score(np.array(y_true)[:, i], y_pred[:, i])
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i], average='weighted')
        precision = precision_score(np.array(y_true)[:, i], y_pred[:, i], average='weighted')
        recall = recall_score(np.array(y_true)[:, i], y_pred[:, i], average='weighted')
        
        df_list.append([col, accuracy, f1, precision, recall])
      
    df = pd.DataFrame(df_list, columns=['Category', 'Accuracy', 'F1_score', 'Precision', 'Recall'])
    
    if sort_by is None:
        return df
    
    df.sort_values(by = sort_by, axis = 0, ascending=False, inplace=True, ignore_index=True)
    return df


def mean_f1_score(y_true, y_pred):
    '''
    Function to calculate the mean f1 score for multi-class classification
    
    Args:
        y_true: a DataFrame with true labels, each row represents the labels for one row in X
        y_pred: 2d array-like with predicted labels, each row represents the labels for one row in X
        
    Returns:
        f1_mean: a float for the mean f1 score
    '''
    f1_list = [f1_score(np.array(y_true)[:, i], y_pred[:, i], average='weighted', 
        labels = np.unique(y_pred), zero_division = 0) for i in range(y_true.shape[1])]
    f1_mean = sum(f1_list) / len(f1_list)
    return f1_mean


def build_model():
    pipeline = Pipeline([
        ('tfidfVect', TfidfVectorizer(tokenizer=tokenize)),
        ('RFclf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
    'tfidfVect__min_df': [1, 5, 10],
    # 'tfidfVect__smooth_idf': [True, False],
    # 'tfidfVect__use_idf': [True, False],
    'RFclf__estimator__bootstrap': [True, False],
    # 'RFclf__estimator__criterion': ['gini', 'entropy'],
    'RFclf__estimator__min_samples_leaf': [1, 2, 5]    
    }

    score_method = make_scorer(mean_f1_score, greater_is_better=True)

    cv = GridSearchCV(pipeline, param_grid=parameters, 
        scoring=score_method, cv=3, verbose=10)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    results = report_metrics(y_test, y_pred)
    for cat in category_names:
        metrics = results[results['Category'] == cat].values.tolist()[0]
        print(cat, ': \n Accuracy: {}, F1_score: {}, Precision {}, Recall: {}'
            .format(metrics[0], metrics[1], metrics[2], metrics[3]))


def save_model(model, model_filepath):
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
    except:
        print('Fail to save the model.')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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