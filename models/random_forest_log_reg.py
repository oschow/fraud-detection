import clean_data as cd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV as GSCV

test_cols = ['gts', 'has_logo']
X_train, X_test, y_train, y_test, X, y = cd.load_train_test(cols=test_cols)

forest = RFC()
random_forest_params = {'n_estimators': [10, 50, 100],
                       'criterion': ['gini', 'entropy'],
                       'max_depth': [10, 100, None],
                       'n_jobs': [-1],
                       'random_state': [42]}

clf = GSCV(forest, random_forest_params)
clf.fit(X, y)

logistic_params = {'penalty': ['l2', 'l1'],
                   'n_jobs': [-1],
                   'random_state': [42]}
