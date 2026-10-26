"""The features, and the agreement between how models and Kaimei encode a passenger."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from titanic_kaimei import CATEGORIES, COLUMNS_V1, COLUMNS_V2, FEATURES, FEATURES_V2, OUTPUTS, encode, load, to_rows
from titanic_kaimei.data import titles


@pytest.fixture(scope="module", params=[1, 2], ids=["v1", "v2"])
def titanic(request):
    return request.param, load(request.param)


def test_every_passenger_has_every_feature(titanic):
    version, data = titanic
    columns = COLUMNS_V1 if version == 1 else COLUMNS_V2
    for frame in (data.passengers, data.new_passengers):
        assert list(frame.columns) == columns
        assert not frame.isna().any().any()
    assert len(data.passengers) == 891 and len(data.new_passengers) == 418
    assert set(data.survived) == {"died", "survived"}


def test_categories_are_sorted_and_cover_the_data(titanic):
    _, data = titanic
    for name, levels in CATEGORIES.items():
        assert levels == sorted(levels)
        seen = set(data.passengers[name]) | set(data.new_passengers[name])
        assert seen <= set(levels), f"{name}: {seen - set(levels)}"


@pytest.mark.parametrize("schema, columns", [(FEATURES, COLUMNS_V1), (FEATURES_V2, COLUMNS_V2)], ids=["v1", "v2"])
def test_schema_matches_the_columns_and_the_encoding(schema, columns):
    """Kaimei encodes a category as its position in the schema's ``categories``; so must the model."""
    assert [feature["name"] for feature in schema] == columns
    for feature in schema:
        if feature["type"] == "categorical":
            assert feature["categories"] == CATEGORIES[feature["name"]]
            assert feature["baseline"] in feature["categories"]
    row = pd.DataFrame([{f["name"]: f["baseline"] for f in schema}])
    encoded = encode(row).iloc[0]
    for feature in schema:
        if feature["type"] == "categorical":
            assert encoded[feature["name"]] == feature["categories"].index(feature["baseline"])
    assert OUTPUTS["positive_label"] in OUTPUTS["labels"]


def test_encode_refuses_unknown_categories():
    row = pd.DataFrame([{"Sex": "unknown", "Embarked": "Southampton", "Title": "Mr"}])
    with pytest.raises(ValueError, match="Unknown Sex"):
        encode(row)


def test_rows_are_plain_json(titanic):
    _, data = titanic
    rows = to_rows(data.passengers.head(20))
    json.dumps(rows)
    assert all(isinstance(value, bool) for row in rows for key, value in row.items() if key in ("IsAlone", "AgeKnown"))


def test_v2_fixes_what_kaimei_found_in_v1():
    v1, v2 = load(1), load(2)
    missing = v1.train_raw["Age"].isna()
    # v1 gives every passenger without an age the same 28 years; v2 flags them and uses the title's median.
    assert set(v1.passengers.loc[missing, "Age"]) == {28.0}
    assert not v2.passengers.loc[missing, "AgeKnown"].any()
    assert v2.passengers.loc[missing, "Age"].nunique() > 1
    # A fare is the price of the whole ticket; v2 divides it among the passengers on the ticket.
    shared = (v2.passengers["TicketSize"] > 1) & (v1.passengers["Fare"] > 0)
    assert (v2.passengers.loc[shared, "FarePerPerson"] < v1.passengers.loc[shared, "Fare"]).all()


def test_titles_group_rare_ones():
    raw = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris", "Rothes, the Countess. of", "Uruchurtu, Don. Manuel E"]})
    assert list(titles(raw)) == ["Mr", "Other", "Other"]
