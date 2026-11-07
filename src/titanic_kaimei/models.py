"""The models of this example, trained the same way in every notebook.

Kaimei runs the forest and the logistic regression itself (it executes uploaded trees, forests,
boosted trees, linear models and scaler pipelines). k-nearest neighbours stands in for a model it
cannot run, because its "weights" are the training data: notebook 04 explains it with locally
computed Shapley values and through a scoring endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Small enough to walk through a tree at a time on the dashboard.
FOREST_PARAMS: Dict[str, Any] = {"n_estimators": 30, "max_depth": 5, "random_state": 0}
KNN_NEIGHBOURS = 15


def make_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**FOREST_PARAMS)


def make_logistic() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])


def make_knn() -> Pipeline:
    """k-nearest neighbours, on scaled features so that Fare doesn't decide who counts as near."""
    return Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=KNN_NEIGHBOURS))])


def shapley_values(
    model: Any, rows: pd.DataFrame, background: pd.DataFrame, *, permutations: int = 8, seed: int = 0
) -> Tuple[np.ndarray, float]:
    """Sampled Shapley values of ``P(survived)``, for a model nothing but its predictions can be asked of.

    Each input is swapped in from a background row, one at a time in random order, and credited with how much the
    probability moved. Returns ``(n_rows, n_columns)`` values and the base value, the model's average over
    ``background``: a row's values add up to its probability minus the base value.
    """
    generator = np.random.default_rng(seed)
    columns = list(rows.columns)
    start = background.to_numpy(dtype=float)
    base = float(model.predict_proba(background)[:, 1].mean())
    values = np.zeros((len(rows), len(columns)))
    for index, row in enumerate(rows.to_numpy(dtype=float)):
        for _ in range(permutations):
            order = generator.permutation(len(columns))
            path = np.repeat(start[None, :, :], len(columns) + 1, axis=0)
            for step, column in enumerate(order, start=1):
                path[step:, :, column] = row[column]
            scored = model.predict_proba(pd.DataFrame(path.reshape(-1, len(columns)), columns=columns))[:, 1]
            means = scored.reshape(len(columns) + 1, -1).mean(axis=1)
            values[index, order] += np.diff(means)
    return values / permutations, base


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
