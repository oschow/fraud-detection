from sklearn.ensemble import GradientBoostingClassifier as GBC
# really annoying to import a module from a dir above/sideways
# http://stackoverflow.com/questions/1054271/how-to-import-a-python-class-that-is-in-a-directory-above
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
import clean_data as cd
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import f1_score, make_scorer
import cPickle as pk
import pandas as pd
from clean_data import prepare_test_dataframe
import json
from clean_data import tokenize

def load_model(filename='../models/GB_model.pk'):
    model = pk.load(open(filename))
    return model

def predict_one(model, data, thresh=0.09189):
    '''
    requires a model object that has a predict_proba method as first arg
    second arg is the data as a matrix or pandas dataframe
    third argument is the threshhold for rounding up

    returns a 1 or 0 (1 == is fraud)
    '''
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    proba = model.predict_proba(data)[0][1]
    if proba >= thresh:
        return 1
    return 0

if __name__ == '__main__':
    model = load_model()
    with open('example.json') as f:
        json_data = json.load(f)
    data = prepare_test_dataframe(json_data)
    prediction = predict_one(model, data)
    if prediction == 1:
        print 'Fraud!'
    else:
        print 'Not Fraud!'
