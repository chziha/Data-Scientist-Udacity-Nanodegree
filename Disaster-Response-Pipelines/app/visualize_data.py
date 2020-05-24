import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import plotly.graph_objs as go
import pandas as pd

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

def return_figures(df):
    '''
    Function to create plotly visualizations

    Args:
        df: a Dataframe containing the data for visualization

    Returns:
        figures: a list contaning plotly visualizations
    '''

    # First figure for the distribution of top 25 most frequent words
    graph_one = []

    words_tot = []
    for _, row in df.iterrows():
        words = tokenize(row['message'])
        words_tot.extend(words)

    word_list, count_list = [], []
    for word in set(words_tot):
        word_list.append(word)
        count_list.append(words_tot.count(word))

    df2 = pd.DataFrame({'word': word_list, 'counts': count_list}).sort_values(
        by='counts', axis=0, ascending=False, inplace=False)

    graph_one.append(
        go.Bar(
            x = df2.iloc[0:25]['word'].tolist(),
            y = df2.iloc[0:25]['counts'].tolist()
            )
        )

    layout_one = dict(title = 'Counts of the Top 25 Most Frequent Words',
        xaxis = dict(title = 'Word', tickangle = 45),
        yaxis = dict(title = 'Counts')
        )

    # Second figure for frequencies of messages per category
    graph_two = []

    message_per_cat = df.iloc[:, 4:].sum().sort_values(ascending = False)

    graph_two.append(
        go.Bar(
            x = message_per_cat.index.tolist(),
            y = message_per_cat.tolist()
            )
        )

    layout_two = dict(title = 'Frequencies of Messages per Category',
        xaxis = dict(title = 'Category', tickangle = 45),
        yaxis = dict(title = 'Frequency')
        )

    # Third figure for frequencies of messages of different genres per category
    graph_three = []

    for genre in df['genre'].unique():
        df_genre = df[df['genre'] == genre].iloc[:, 4:].sum()
        graph_three.append(go.Bar(
            x = df_genre.index.tolist(),
            y = df_genre.tolist(),
            name = genre
            )
        )

    layout_three = dict(title = 'Frequencies of Messages of different genres per Category',
        xaxis = dict(title = 'Category', tickangle = 45),
        yaxis = dict(title = 'Frequency'),
        barmode = 'stack'
        )

    # Append the plotly visualizations into the list
    figures = []
    figures.append(dict(data = graph_one, layout = layout_one))
    figures.append(dict(data = graph_two, layout = layout_two))
    figures.append(dict(data = graph_three, layout = layout_three))

    return figures