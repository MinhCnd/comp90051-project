import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier


def simple_logistic_regression(X_train, y_train, lambda_val):
    
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        loss='log_loss',
    ) 

    return gradient_boosting.fit(X_train, y_train)


def simple_logistic_regression_results(X_train, y_train, X_test, lambda_val):
    logreg = simple_logistic_regression(X_train, y_train, lambda_val)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)   # both columns, not just [:, 1]

    results = np.concatenate(
        (y_pred_proba, y_pred.reshape(-1, 1)),
        axis=1
    )
    return results
