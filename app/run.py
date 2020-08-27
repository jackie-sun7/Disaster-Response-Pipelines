import json
import plotly
import pandas as pd

import nltk
nltk.download(['wordnet', 'punkt','stopwords', 'averaged_perceptron_tagger'])

import sys
import pandas as pd
import numpy as np
import re
import sklearn
import pickle


from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


# functions in model
def tokenize(text):
    """
    Word tokenize message
    Remove stopwords and special characters
    Lemmatize token according to part of speech
    """
    
    # initialize part of speech dictionary
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    sentences = sent_tokenize(text)
    
    for sent in sentences:
        sent = re.sub("[^a-zA-Z0-9]", " ", sent)
        tokens = word_tokenize(sent)
        # get each token's part of speech
        pos = [tag_dict.get(x[1][0],wordnet.NOUN) for x in nltk.pos_tag(tokens)] 
    

        for i, tok in enumerate(tokens):
            # lemmatize token according to part of speech
            clean_tok = lemmatizer.lemmatize(tok, pos[i])

            if clean_tok not in stopwords.words("english"):
                clean_tokens.append(clean_tok.lower())
            
    return clean_tokens


def get_msg_cols(df):
    """
    To use nlp pipeline for messgae columns, save all message in an array
    """
    
    return [msg for msg in df.message]

def get_genre_cols(df):
    """
    To use onehotencode pipeline for genre columns, get genre column
    """
    
    return df[['genre']]


class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)


class ClfSwitcher(BaseEstimator):
    """
    Switch Classifier in pipeline
    """
    
    def __init__(self, estimator = RandomForestClassifier()):   
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
  
    def score(self, X, y):
        return self.estimator.score(X, y)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")



df_direct = pd.read_pickle('dirct_cnt.plk').iloc[:100,:]
df_social = pd.read_pickle('social_cnt.plk').iloc[:100,:]
df_news = pd.read_pickle('news_cnt.plk').iloc[:100,:]


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
               {
            'data': [
                Bar(
                    x=df_direct.index,
                    y=df_direct.frequency
                )
            ],

            'layout': {
                'title': 'Distribution of Word Frequency in Direct Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': 45
                }
            }
        },
                       {
            'data': [
                Bar(
                    x=df_social.index,
                    y=df_social.frequency
                )
            ],

            'layout': {
                'title': 'Distribution of Word Frequency in Social Media',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': 45
                }
            }
        },
                       {
            'data': [
                Bar(
                    x=df_news.index,
                    y=df_news.frequency
                )
            ],

            'layout': {
                'title': 'Distribution of Word Frequency in News',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': 45
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    genre = request.args.get('genre', '')

    # use model to predict classification for query
    
    classification_labels = model.predict(pd.DataFrame({'message' : [query] ,'genre' : [genre]}))[0]
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
