import sys
import os.path
sys.path.append(
   os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir)))
import clean_data as cd
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# loading in data
df = cd.load_clean_df()
df.head()

# statsmodels logit
X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test()
X_const = add_constant(X_train, prepend = True)
logit_model = Logit(y_train, X_const).fit()
logit_model.summary()

# sklearn LogisticRegression
accuracies = []
precisions = []
recalls = []
model = LogisticRegression()
LR_model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_true = y_test
accuracies.append(accuracy_score(y_true,y_predict))
precisions.append(precision_score(y_true,y_predict))
recalls.append(recall_score(y_true,y_predict))
print "accuracy: ", np.average(accuracies)
print "precision: ", np.average(precisions)
print "recall: ", np.average(recalls)
