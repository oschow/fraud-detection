import clean_data as cd
from sklearn.ensemble import RandomForestClassifier as RFC
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import roc_auc_score
import cPickle as pk

def optimize_RF(cols='all', light=False):
    '''
    optimizes random forest model
    '''
    if cols is None:
        cols = ['gts', 'has_logo']
    elif cols=='all':
        cols = None
    X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test(cols=cols)

    forest = RFC()

    if light:
        random_forest_params = {'n_estimators': [100, 200],
                               'max_depth': [5, 10, 15, 20],
                               'n_jobs': [-1],
                               'random_state': [42]}
    else:
        random_forest_params = {'n_estimators': [50, 75, 100, 200, 300, 400],
                               'criterion': ['gini', 'entropy'],
                               'max_depth': [5, 10, 15, 20, None],
                               'n_jobs': [-1],
                               'random_state': [42]}
        '''
        {'criterion': 'gini',
         'max_depth': 10,
         'n_estimators': 200,
         'n_jobs': -1,
         'random_state': 42}
        '''

    clf = GSCV(forest, random_forest_params)
    clf.fit(X, y)
    print 'best params:'
    clf.best_params_
    return clf

def fit_model(train_test=True, params=None):
    '''
    uses ideal params from initial gridsearch on simplified columns by default
    '''
    if params is None:
        params = {'criterion': 'gini',
                 'max_depth': 10,
                 'n_estimators': 200,
                 'n_jobs': -1,
                 'random_state': 42}

    X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test()
    forest = RFC(**params)
    if train_test:
        forest.fit(X_train, y_train)
    else:
        forest.fit(X, y)

    return forest

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

def pickle_model(model, filename='RF_model.pk'):
    pk.dump(model, open(filename, 'w'), 2)

def unpickle_and_predict(raw_json_data, filename='RF_model.pk'):
    '''
    does what the function says, unpickles a model
    and makes predictions on the raw json data
    the raw json should be of the same format as data/data.json
    '''
    model = pk.load(open(filename))
    df = pd.read_json(raw_json_data)
    df = cd.prepare_test_dataframe(df)
    predictions = model.predict_proba(df.values)
    return predictions
