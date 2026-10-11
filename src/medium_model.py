import numpy as np
from xgboost import XGBClassifier

def medium_gradient_boosting(X_train, y_train, scale_pos_weight=1.0):
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    return clf.fit(X_train, y_train)


def medium_gradient_boosting_results(X_train, y_train, X_test, scale_pos_weight=1.0):
    clf = medium_gradient_boosting(X_train, y_train, scale_pos_weight)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    return np.concatenate(
        (y_pred_proba, y_pred.reshape(-1, 1)),
        axis=1
    )