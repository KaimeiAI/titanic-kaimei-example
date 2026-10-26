"""The models of this example, trained the same way in every notebook.

Kaimei runs the forest and the logistic regression itself (it executes uploaded trees, forests of
up to 30 trees, linear models and scaler pipelines). XGBoost stands in for a model it cannot run:
notebook 04 explains it with locally computed SHAP values and through a scoring endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Kaimei executes a forest of up to 30 trees and 4,000 nodes; depth 5 keeps it near 1,500.
FOREST_PARAMS: Dict[str, Any] = {"n_estimators": 30, "max_depth": 5, "random_state": 0}
XGBOOST_PARAMS: Dict[str, Any] = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "random_state": 0}


def make_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**FOREST_PARAMS)


def make_logistic() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])


def make_xgboost() -> Any:
    """An XGBoost classifier (``pip install xgboost``). Trained on 0/1 labels, like XGBoost wants."""
    from xgboost import XGBClassifier

    return XGBClassifier(**XGBOOST_PARAMS)


def split(X: pd.DataFrame, y: pd.Series):
    """The 80/20 split every notebook validates on (stratified, seed 0)."""
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def scores(models: Mapping[str, Any], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """5-fold cross-validated accuracy and accuracy on the held-out 20%, one row per model.

    Each model is left fitted on the 80% training part of :func:`split`.
    """
    X_train, X_valid, y_train, y_valid = split(X, y)
    rows = {}
    for name, model in models.items():
        cv = cross_val_score(model, X, y, cv=5).mean()
        model.fit(X_train, y_train)
        rows[name] = {"5-fold CV accuracy": cv, "validation accuracy": model.score(X_valid, y_valid)}
    return pd.DataFrame(rows).T
