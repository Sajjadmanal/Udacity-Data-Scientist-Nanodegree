# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
#nltk.download()
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
nltk.download('stopwords')
from sklearn.metrics import fbeta_score, make_scorer
import pickle
import sys


def f1_pre_acc_evaluation (y_true, y_pred):
    """A function that measures mean of f1, precision, recall for each class within multi-class prediction
       Returns a dataframe with columns:
       f1-score (average for all possible values of specific class)
       precision (average for all possible values of specific class)
       recall (average for all possible values of specific class)
       kindly keep in mind that some classes might be imbalanced and average values may mislead.
    """
    #instantiating a dataframe
    report = pd.DataFrame ()

    for col in y_true.columns:
        #returning dictionary from classification report
        class_dict = classification_report (output_dict = True, y_true = y_true.loc [:,col], y_pred = y_pred.loc [:,col])

        #converting from dictionary to dataframe
        eval_df = pd.DataFrame (pd.DataFrame.from_dict (class_dict))

       # print (eval_df)

        #dropping unnecessary columns
        eval_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis =1, inplace = True)

        #dropping unnecessary row "support"
        eval_df.drop(index = 'support', inplace = True)

        #calculating mean values
        av_eval_df = pd.DataFrame (eval_df.transpose ().mean ())

        #transposing columns to rows and vice versa
        av_eval_df = av_eval_df.transpose ()

        #appending result to report df
        report = report.append (av_eval_df, ignore_index = True)

    #renaming indexes for convinience
    report.index = y_true.columns

    return report

def f1_scorer_eval (y_true, y_pred):
    """A function that measures mean of F1 for all classes
       Returns an average value of F1 for sake of evaluation whether model predicts better or worse in GridSearchCV
    """
    #converting y_pred from np.array to pd.dataframe
    #keep in mind that y_pred should a pd.dataframe rather than np.array
    y_pred = pd.DataFrame (y_pred, columns = y_true.columns)


    #instantiating a dataframe
    report = pd.DataFrame ()

    for col in y_true.columns:
        #returning dictionary from classification report
        class_dict = classification_report (output_dict = True, y_true = y_true.loc [:,col], y_pred = y_pred.loc [:,col])

        #converting from dictionary to dataframe
        eval_df = pd.DataFrame (pd.DataFrame.from_dict (class_dict))

        #dropping unnecessary columns
        eval_df.drop(['micro avg', 'macro avg', 'weighted avg'], axis =1, inplace = True)

        #dropping unnecessary row "support"
        eval_df.drop(index = 'support', inplace = True)

        #calculating mean values
        av_eval_df = pd.DataFrame (eval_df.transpose ().mean ())

        #transposing columns to rows and vice versa
        av_eval_df = av_eval_df.transpose ()

        #appending result to report df
        report = report.append (av_eval_df, ignore_index = True)

    #returining mean value for all classes. since it's used for GridSearch we may use mean
    #as the overall value of F1 should grow.
    return report ['f1-score'].mean ()


def load_data(database_filepath):
    """
    Function that loads messages and categories from database using database_filepath as a filepath and sqlalchemy as library
    Returns two dataframes X and y

    """
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql ('SELECT * FROM MessagesCategories', engine)
    X = df ['message']
    y = df.iloc[:,4:]

    return X, y

def tokenize(text):
    """Tokenization function. Receives as input raw text which afterwards normalized, stop words removed, stemmed and lemmatized.
    Returns tokenized text"""

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")

    #tokenize
    words = word_tokenize (text)

    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]

    return words_lemmed


def build_model():

    #setting pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
        ])

    # fbeta_score scoring object using make_scorer()
    scorer = make_scorer (f1_scorer_eval)

    #model parameters for GridSearchCV
    parameters = {  'vect__max_df': (0.75, 1.0),
                    'clf__estimator__n_estimators': [10, 20],
                    'clf__estimator__min_samples_split': [2, 5]
              }
    cv = GridSearchCV (pipeline, param_grid= parameters, scoring = scorer, verbose =7 )

    return cv
    #return pipeline

def evaluate_model(model, X_test, Y_test):
    y_pred_tuned = model.predict (X_test)
    #converting to a dataframe
    y_pred_tuned = pd.DataFrame (y_pred_tuned, columns = Y_test.columns)

    report_tuned = f1_pre_acc_evaluation (Y_test, y_pred_tuned)

    #display result of model evaluation
    print (report_tuned)

def save_model(model, model_filepath):
    """ Saving model's best_estimator_ using pickle
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #X, Y, category_names = load_data(database_filepath)
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
