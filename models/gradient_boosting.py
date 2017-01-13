from sklearn.ensemble import GradientBoostingClassifier as GBC
import clean_data as cd
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import f1_score, make_scorer
import cPickle as pk

def optimize_GBC(cols='all', light=False):
    '''
    optimizes random forest model
    '''
    if cols is None:
        cols = ['gts', 'has_logo']
    elif cols=='all':
        cols = None
    X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test(cols=cols)

    gb = GBC()

    if light:
        pass
    else:
        gb_params = {'n_estimators': [500],
                    'learning_rate': [0.02, 0.05, 0.08, 0.1],
                    'subsample': [0.3, 0.4, 0.5],
                    'max_features': ['sqrt', None],
                    'random_state': [42]}
        '''
        best found:
        {'learning_rate': 0.08,
         'max_features': 'sqrt',
         'n_estimators': 500,
         'random_state': 42,
         'subsample': 0.5}
        '''

    clf = GSCV(gb, gb_params, scoring=make_scorer(f1_score), n_jobs=-1)
    clf.fit(X, y)
    print 'best params:'
    print clf.best_params_
    return clf

def fit_model(train_test=True, params=None):
    '''
    uses ideal params from initial gridsearch on simplified columns by default
    '''
    if params is None:
        params = {'learning_rate': 0.08,
                 'max_features': 'sqrt',
                 'n_estimators': 500,
                 'random_state': 42,
                 'subsample': 0.5}

    X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test()
    gb = GBC(**params)
    if train_test:
        gb.fit(X_train, y_train)
    else:
        gb.fit(X, y)

    return gb

def predict_test(model=None):
    '''
    example use:
    y_pred, model = predict_test()

    will predict probabilities of fraud (1) or not (0)
    and return that array and the model (in case it was made
    by default)
    '''
    if model is None:
        model = fit_model()

    X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test()
    y_pred = model.predict_proba(X_test)
    return y_pred, model

def pickle_model(model, filename='GB_model.pk'):
    pk.dump(model, open(filename, 'w'), 2)

def unpickle_and_predict(raw_json_data, filename='GB_model.pk'):
    '''
    does what the function says, unpickles a model
    and makes predictions on the raw json data
    the raw json should be of the same format as data/data.json
    '''
    model = pk.load(open(filename))
    df = cd.prepare_test_dataframe(raw_json_data)
    predictions = model.predict_proba(df)
    return predictions
