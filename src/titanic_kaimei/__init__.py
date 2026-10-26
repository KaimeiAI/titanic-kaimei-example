"""Shared code for the Titanic × Kaimei example notebooks.

* :mod:`titanic_kaimei.data` loads the Kaggle files and builds the v1 / v2 features.
* :mod:`titanic_kaimei.schema` holds the Kaimei feature schemas.
* :mod:`titanic_kaimei.models` trains the forest, logistic regression and XGBoost models.
* :mod:`titanic_kaimei.workspace` connects to Kaimei and keeps re-runs idempotent.
* :mod:`titanic_kaimei.serving` is a scoring service for Kaimei's remote inference.
"""

from .data import CATEGORIES, COLUMNS_V1, COLUMNS_V2, Titanic, encode, load, to_row, to_rows
from .schema import FEATURES, FEATURES_V2, OUTPUTS

__all__ = [
    "CATEGORIES",
    "COLUMNS_V1",
    "COLUMNS_V2",
    "FEATURES",
    "FEATURES_V2",
    "OUTPUTS",
    "Titanic",
    "encode",
    "load",
    "to_row",
    "to_rows",
]
__version__ = "1.0.0"
