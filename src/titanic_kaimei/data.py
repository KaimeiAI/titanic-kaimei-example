"""Load the Kaggle Titanic files and turn them into model features.

Two feature sets:

* **v1** is the original Kaggle notebook's feature engineering (median age, family size,
  travelling alone, title), with categories kept as words so Kaimei can show them as words.
* **v2** fixes what Kaimei's analysis of v1 found (notebook 03): the fare is divided among
  everyone on the ticket, a missing age is filled from the passenger's title and flagged,
  and the derived family features are replaced by the size of the travelling party.

Models never see words: :func:`encode` turns categories into their position in
:data:`CATEGORIES`, which is exactly what Kaimei's feature schema does with the same lists.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

#: Where the CSVs are fetched from when this package runs outside a clone (e.g. on Colab).
DATA_URL = "https://raw.githubusercontent.com/KaimeiAI/titanic-kaimei-example/main/data"

PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
#: Titles with a handful of passengers each, grouped as "Other" (as in the original notebook).
RARE_TITLES = ["Mlle", "Major", "Col", "Sir", "Don", "Mme", "Jonkheer", "Lady", "Capt", "Countess", "Ms", "Dona"]

#: Every categorical feature's levels, sorted. A model sees a level as its index in this list,
#: and the Kaimei feature schema lists the same levels, so both encode a row identically.
CATEGORIES: Dict[str, List[str]] = {
    "Sex": ["female", "male"],
    "Embarked": ["Cherbourg", "Queenstown", "Southampton"],
    "Title": ["Dr", "Master", "Miss", "Mr", "Mrs", "Other", "Rev"],
}

COLUMNS_V1 = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamSize", "IsAlone", "Title"]
COLUMNS_V2 = ["Pclass", "Sex", "Age", "AgeKnown", "SibSp", "Parch", "FarePerPerson", "TicketSize", "Embarked", "Title"]
BOOLEAN_FEATURES = ("IsAlone", "AgeKnown")


def data_dir() -> Path:
    """The folder holding ``train.csv`` and ``test.csv``.

    ``TITANIC_DATA_DIR`` wins; then the ``data/`` folder of this repository; otherwise the files
    are downloaded once from GitHub into ``~/.cache/titanic-kaimei``.
    """
    configured = os.environ.get("TITANIC_DATA_DIR")
    if configured:
        return Path(configured)
    repo = Path(__file__).resolve().parents[2] / "data"
    if (repo / "train.csv").exists():
        return repo
    cache = Path.home() / ".cache" / "titanic-kaimei"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv"):
        target = cache / name
        if not target.exists():
            urllib.request.urlretrieve(f"{DATA_URL}/{name}", target)  # noqa: S310 - fixed https URL
    return cache


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kaggle's ``train.csv`` (891 passengers, with ``Survived``) and ``test.csv`` (418, without)."""
    folder = data_dir()
    return pd.read_csv(folder / "train.csv"), pd.read_csv(folder / "test.csv")


def titles(raw: pd.DataFrame) -> pd.Series:
    """``Mr``, ``Mrs``, ``Miss``, ... taken from the name; rare titles become ``Other``."""
    return raw["Name"].str.extract(r"([A-Za-z]+)\.", expand=False).replace(RARE_TITLES, "Other")


def prepare(raw: pd.DataFrame, train_raw: pd.DataFrame) -> pd.DataFrame:
    """v1 features, as in the original Kaggle notebook: medians come from ``train_raw``."""
    df = pd.DataFrame(
        {
            "Pclass": raw["Pclass"],
            "Sex": raw["Sex"],
            "Age": raw["Age"].fillna(train_raw["Age"].median()),
            "SibSp": raw["SibSp"],
            "Parch": raw["Parch"],
            "Fare": raw["Fare"].fillna(train_raw["Fare"].median()).round(2),
            "Embarked": raw["Embarked"].fillna("S").map(PORTS),
        }
    )
    df["FamSize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = df["FamSize"] == 1
    df["Title"] = titles(raw)
    return df[COLUMNS_V1]


def prepare_v2(raw: pd.DataFrame, train_raw: pd.DataFrame, all_raw: pd.DataFrame) -> pd.DataFrame:
    """v2 features: fare per person on the ticket, age filled from the title (and flagged).

    ``all_raw`` is every passenger whose ticket counts toward ``TicketSize`` (train and test:
    in production this is the booking, which knows how many people share a ticket).
    """
    df = prepare(raw, train_raw)
    title = titles(raw)
    title_age = train_raw.assign(Title=titles(train_raw)).groupby("Title")["Age"].median()
    df["AgeKnown"] = raw["Age"].notna()
    df["Age"] = raw["Age"].fillna(title.map(title_age)).round(1)
    df["TicketSize"] = raw["Ticket"].map(all_raw.groupby("Ticket")["PassengerId"].count()).fillna(1).astype(int)
    df["FarePerPerson"] = (df["Fare"] / df["TicketSize"]).round(2)
    return df[COLUMNS_V2]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Numbers a model can train on: categories become their index in :data:`CATEGORIES`."""
    out = df.copy()
    for name, levels in CATEGORIES.items():
        if name in out:
            codes = out[name].map({level: i for i, level in enumerate(levels)})
            if codes.isna().any():
                unknown = sorted(set(out.loc[codes.isna(), name].astype(str)))
                raise ValueError(f"Unknown {name} values {unknown}; expected one of {levels}.")
            out[name] = codes
    for name in BOOLEAN_FEATURES:
        if name in out:
            out[name] = out[name].astype(int)
    return out.astype(float)


def to_row(values: Mapping[str, Any]) -> Dict[str, Any]:
    """One passenger as JSON-ready values (NumPy scalars become Python numbers and booleans)."""
    row: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (bool, np.bool_)):
            row[key] = bool(value)
        elif hasattr(value, "item"):
            row[key] = value.item()
        else:
            row[key] = value
    return row


def to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Every row of ``df`` as :func:`to_row` dicts, ready to send to Kaimei."""
    return [to_row(record) for record in df.to_dict("records")]


@dataclass
class Titanic:
    """The data every notebook starts from."""

    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    passengers: pd.DataFrame
    """Features of the 891 passengers with known outcomes."""
    survived: pd.Series
    """``"died"`` or ``"survived"`` for each of them."""
    new_passengers: pd.DataFrame
    """Features of the 418 passengers in Kaggle's test file (no outcome): stand-ins for new traffic."""


def load(version: int = 1) -> Titanic:
    """Load and prepare the data with the v1 (original) or v2 (improved) features."""
    train_raw, test_raw = load_raw()
    if version == 1:
        passengers, new_passengers = prepare(train_raw, train_raw), prepare(test_raw, train_raw)
    elif version == 2:
        all_raw = pd.concat([train_raw, test_raw], ignore_index=True)
        passengers = prepare_v2(train_raw, train_raw, all_raw)
        new_passengers = prepare_v2(test_raw, train_raw, all_raw)
    else:
        raise ValueError("version must be 1 or 2")
    survived = train_raw["Survived"].map({0: "died", 1: "survived"}).rename("Survived")
    return Titanic(train_raw, test_raw, passengers, survived, new_passengers)
