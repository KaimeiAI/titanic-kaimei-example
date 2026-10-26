"""The scoring service answers in the shape Kaimei's remote inference expects."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from titanic_kaimei import encode, load, to_rows  # noqa: E402
from titanic_kaimei.models import make_logistic  # noqa: E402
from titanic_kaimei.serving import create_app  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    data = load(1)
    model = make_logistic().fit(encode(data.passengers), data.survived)
    return model, data


def test_predict_returns_one_probability_pair_per_row(setup):
    model, data = setup
    client = TestClient(create_app(model))
    rows = to_rows(data.new_passengers.head(5))
    response = client.post("/predict", json={"instances": rows})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 5
    expected = model.predict_proba(encode(data.new_passengers.head(5)))
    for got, want in zip(predictions, expected):
        assert got == pytest.approx(list(want))
        assert sum(got) == pytest.approx(1.0)


def test_bad_rows_are_rejected(setup):
    model, data = setup
    client = TestClient(create_app(model))
    row = to_rows(data.new_passengers.head(1))[0]
    assert client.post("/predict", json={"instances": [{**row, "Sex": "unknown"}]}).status_code == 422
    del row["Fare"]
    assert client.post("/predict", json={"instances": [row]}).status_code == 422


def test_optional_bearer_token(setup):
    model, data = setup
    client = TestClient(create_app(model, api_key="s3cret"))
    body = {"instances": to_rows(data.new_passengers.head(1))}
    assert client.post("/predict", json=body).status_code == 401
    assert client.post("/predict", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
