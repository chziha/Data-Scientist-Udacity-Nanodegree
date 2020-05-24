import re
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer
from sqlalchemy import create_engine

from visualize_data import return_figures


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

graphs = return_figures(df)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()