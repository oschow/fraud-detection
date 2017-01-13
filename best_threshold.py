import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import confusion_matrix, roc_auc_score
import clean_data as cd
from models.random_forest import fit_model as rf_fit
from models.gradient_boosting import fit_model as gb_fit


def profit_curve(cost_benefit_matrix, probabilities, y_true):
    thresholds = sorted(probabilities)
    thresholds.append(1.0)
    thresholds = thresholds[::-1]
    profits = []
    for threshold in thresholds:
        y_predict = probabilities >= threshold
        confusion_mat = confusion_matrix(y_true, y_predict)
        profit = np.sum(confusion_mat * cost_benefit_matrix) / float(len(y_true))
        profits.append(profit)
    return thresholds, profits

def run_profit_curve(model, costbenefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    thresholds, profits = profit_curve(costbenefit, y_pred, y_test)
    return thresholds, profits

def plot_profit_models(models, costbenefit, X_train, X_test, y_train, y_test):
    percentages = np.linspace(0, 100, len(y_test) + 1)
    for model in models:
        thresholds, profits = run_profit_curve(model,
                                               costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        plt.plot(percentages, profits, label=model.__class__.__name__)
    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='upper right')
    plt.savefig('profit_curve.png')
    plt.show()

def find_best_threshold(models, costbenefit, X_train, X_test, y_train, y_test):
    max_model = None
    max_threshold = None
    max_profit = None
    for model in models:
        thresholds, profits = run_profit_curve(model, costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model.__class__.__name__
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit

def roc_curve(probabilities, y_true):
    threshs = np.sort(probabilities)
    tprs = []
    fprs = []
    num_positive_cases = sum(y_true)
    num_negative_cases = len(y_true) - num_positive_cases

    for thresh in threshs:
        y_pred = probabilities >= thresh
        true_positives = np.sum(y_pred * y_true)
        false_positives = np.sum(y_pred) - true_positives
        tpr = true_positives / float(num_positive_cases)
        fpr = false_positives / float(num_negative_cases)
        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, threshs.tolist()

def run_roc_curve(model, costbenefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    tprs, fprs, threshs = roc_curve(y_pred, y_test)
    print '{} auc score:'.format(model.__class__.__name__)
    print roc_auc_score(y_test, y_pred)
    return tprs, fprs, threshs

def plot_roc_curve(models, costbenefit, X_train, X_test, y_train, y_test):
    for model in models:
        tprs, fprs, threshs = run_roc_curve(model,
                                               costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)

        plt.plot(fprs, tprs, label=model.__class__.__name__)
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.legend(loc='upper right')
    plt.savefig('roc_curve.png')
    plt.show()

def plot_confusion_matrix(gradient_boost, max_threshold, X_train, X_test, y_train, y_test):
    gradient_boost.fit(X_train, y_train)
    y_predicted = gradient_boost.predict_proba(X_test)[:, 1]
    y_pre = y_predicted >= max_threshold
    confusion_mat = confusion_matrix(y_test, y_pre)
    print confusion_mat
    plt.matshow(confusion_mat)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_feature_importance(gradient_boost, columns):
    features = gradient_boost.feature_importances_
    feature_importance = 100.0 * (features / features.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, columns[sorted_idx])
    plt.xlim([0, 110])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, y, columns = cd.load_train_test()
    costbenefit = np.array([[0, -10], [-100, 0]])
    forest = rf_fit()
    gradient_boost = gb_fit()
    models = [forest, LR(), gradient_boost]
    plot_profit_models(models, costbenefit, X_train, X_test, y_train, y_test)
    plot_roc_curve(models, costbenefit, X_train, X_test, y_train, y_test)
    max_model, max_threshold, max_profit = find_best_threshold(models, costbenefit, X_train, X_test, y_train, y_test)
    print 'Best Model, Best Threshold, Minimum Loss:'
    print max_model, max_threshold, max_profit
    print np.array([[4130, 175],[31, 396]])
    plot_confusion_matrix(gradient_boost, max_threshold, X_train, X_test, y_train, y_test)
    plot_feature_importance(gradient_boost, columns)
