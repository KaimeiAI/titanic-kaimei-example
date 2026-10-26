"""Kaimei feature schemas for the two feature sets.

The schema is the contract between your data and Kaimei: it says how to read a passenger
(``type`` and, for categories, the ``categories`` in the order the model was encoded with), how to
show it (``unit``, ``description``) and what a typical passenger looks like (``baseline``, which
pre-fills the dashboard's Explain form).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .data import CATEGORIES

#: What the models predict, and which class an explanation explains by default.
OUTPUTS: Dict[str, Any] = {"labels": ["died", "survived"], "positive_label": "survived"}

FEATURES: List[Dict[str, Any]] = [
    {"name": "Pclass", "type": "integer", "min": 1, "max": 3, "baseline": 3,
     "description": "Ticket class: 1 = first, 2 = second, 3 = third"},
    {"name": "Sex", "type": "categorical", "categories": CATEGORIES["Sex"], "baseline": "male"},
    {"name": "Age", "type": "numeric", "unit": "years", "min": 0, "max": 80, "baseline": 28},
    {"name": "SibSp", "type": "integer", "min": 0, "max": 8, "baseline": 0,
     "description": "Siblings and spouses aboard"},
    {"name": "Parch", "type": "integer", "min": 0, "max": 6, "baseline": 0,
     "description": "Parents and children aboard"},
    {"name": "Fare", "type": "numeric", "unit": "GBP", "min": 0, "max": 512, "baseline": 14.45,
     "description": "Price of the whole ticket, shared by everyone on it"},
    {"name": "Embarked", "type": "categorical", "categories": CATEGORIES["Embarked"], "baseline": "Southampton",
     "description": "Port of embarkation"},
    {"name": "FamSize", "type": "integer", "min": 1, "max": 11, "baseline": 1,
     "description": "Family members aboard, including the passenger (SibSp + Parch + 1)"},
    {"name": "IsAlone", "type": "binary", "baseline": True, "description": "Travelling without family"},
    {"name": "Title", "type": "categorical", "categories": CATEGORIES["Title"], "baseline": "Mr",
     "description": "Title from the passenger's name; rare titles are grouped as Other"},
]

FEATURES_V2: List[Dict[str, Any]] = [
    FEATURES[0],
    FEATURES[1],
    {"name": "Age", "type": "numeric", "unit": "years", "min": 0, "max": 80, "baseline": 30,
     "description": "Missing ages are filled with the median age of the passenger's title"},
    {"name": "AgeKnown", "type": "binary", "baseline": True, "description": "The passenger list records an age"},
    FEATURES[3],
    FEATURES[4],
    {"name": "FarePerPerson", "type": "numeric", "unit": "GBP", "min": 0, "max": 512, "baseline": 8.05,
     "description": "Ticket price divided by the number of passengers on the ticket"},
    {"name": "TicketSize", "type": "integer", "min": 1, "max": 11, "baseline": 1,
     "description": "Passengers travelling on the same ticket"},
    FEATURES[6],
    FEATURES[9],
]
