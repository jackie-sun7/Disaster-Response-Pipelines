# import libraries
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


def load_data(database_filepath):
    """
    Load date from SQL
    Split dependent and independent variables
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()
    df = pd.read_sql("SELECT * FROM DisasterResponse ", conn)
    
    # load message and genre as independent variables, others as dependent variables
    X = df.iloc[:, [1,3]]
    Y = df.iloc[:,4:]
    return X, Y, Y.columns


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
        sent = re.sub("[^a-zA-Z]", " ", sent)
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
    
    
def build_model():
    """
    Set nlp and onehotencode parallel pipeline with MultiOutputClassifier
    Prepare candidate model
    """
    
    #parallel tranform message and genre column
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('msg_pipeline', Pipeline([
                ('get_msg_col', FunctionTransformer(get_msg_cols, validate=False)),
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('genre_pipeline', Pipeline([
                ('get_genre_col', FunctionTransformer(get_genre_cols, validate=False)),
                ('label', ModifiedLabelEncoder()),
                ('ohe', OneHotEncoder())
            ]))
        ])),
         ('clf', MultiOutputClassifier(ClfSwitcher()))
    ])
    
    #Set XGBoost and RandomForest as candidate model
    parameters = [
          {
          'clf__estimator__estimator': [GradientBoostingClassifier()],
          'clf__estimator__estimator__n_estimators': [50, 100, 200]
          },
          {
          'clf__estimator__estimator': [RandomForestClassifier()],
          'clf__estimator__estimator__n_estimators': [50, 100, 200]
          }
        ]
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Report the f1 score, precision and recall for each output category of the dataset
    """
    
    Y_pred = model.predict(X_test)
           
    for i, label in enumerate(category_names):
        scores = sklearn.metrics.classification_report(Y_test.iloc[:,i], Y_pred[:,i])
        accuracy = (Y_pred[:,i] == Y_test.iloc[:,i]).mean()

        print('\033[1m', f"\n{label.capitalize() }:\n",
              '\033[0m', scores)
        print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
        
def frequency(df, genre, path):
    """
    Get word frenquency for each genre and save in pickle
    """
    
    sub_df = df[df['genre'] == genre]
    cv = CountVectorizer(tokenizer=tokenize)
    cv_array = cv.fit_transform(sub_df['message'].values.ravel())
    cv_dict = pd.DataFrame(
        {'frequency': cv_array.toarray().sum(axis=0)},
        index = cv.get_feature_names()
    )
    cv_dict.sort_values(by='frequency', ascending=False, inplace=True)
    cv_dict.to_pickle(path)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Get word frenquency for each genre...')
        frequency(X, 'direct', '../app/dirct_cnt.plk')
        frequency(X, 'social', '../app/social_cnt.plk')
        frequency(X, 'news', '../app/news_cnt.plk')
        
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
