import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def simple_logistic_regression(X_train, y_train, lambda_val):
    logreg = LogisticRegression(penalty='l1', C=1/lambda_val, solver='liblinear', max_iter=1000)
    logreg = logreg.fit(X_train, y_train)

    return logreg


def simple_logistic_regression_results(X_train, y_train, X_test, C_val):
    logreg = simple_logistic_regression(X_train, y_train, C_val)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]
    results = np.concatenate((y_pred.reshape(-1, 1), y_pred_proba.reshape(-1, 1)), axis=1)
    return results
