import sys
import pandas as pd
import numpy as np

import pickle

from sqlalchemy import create_engine

import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')

from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV,cross_val_score, cross_validate
from sklearn.metrics import fbeta_score, make_scorer, SCORERS



def load_data(database_filepath):
    table_name = 'disaster_response'
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = pd.read_sql_table(table_name, engine)
    X =  df['message'].values
    Y = df.iloc[:, 4:].values

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords.words('english')]
    stem_text = [PorterStemmer().stem(w) for w in text]
    
    lem_text = [WordNetLemmatizer().lemmatize(w, pos='v') for w in stem_text]
    

    return lem_text
    


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    
        ])
    
    parameters = {
        
        'clf__estimator__oob_score': [True],
        'clf__estimator__min_samples_split': [2, 3]
        
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv



def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()
    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test[0],                                 grid_fit.best_estimator_.predict(X_test) [0], beta = 0.5, average = "micro")))
    print("\nBest Parameters:", cv.best_params_)
    


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict()
    display_result(cv,y_test,y_pred)

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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