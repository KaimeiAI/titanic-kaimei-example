"""The models train, stay accurate, and stay within what Kaimei can run."""

from __future__ import annotations

import pytest
from kaimei import StructureError
from kaimei.structure import extract_structure, spec_is_executable

from titanic_kaimei import COLUMNS_V1, COLUMNS_V2, encode, load
from titanic_kaimei.models import FOREST_PARAMS, make_forest, make_logistic, scores, split


@pytest.fixture(scope="module")
def v1():
    data = load(1)
    return encode(data.passengers), data.survived


def test_forest_fits_kaimei_execution_limits(v1):
    X, y = v1
    forest = make_forest().fit(*split(X, y)[::2])
    nodes = sum(tree.tree_.node_count for tree in forest.estimators_)
    assert FOREST_PARAMS["n_estimators"] <= 30 and nodes <= 4000
    spec = extract_structure(forest, feature_names=COLUMNS_V1).spec
    assert spec["family"] == "ensemble" and len(spec["trees"]) == 30
    assert spec_is_executable(spec)


def test_logistic_uploads_with_its_scaler(v1):
    X, y = v1
    spec = extract_structure(make_logistic().fit(X, y), feature_names=COLUMNS_V1).spec
    assert [layer["type"] for layer in spec["layers"]] == ["input", "standardize", "dense"]
    assert spec_is_executable(spec)


def test_knn_cant_be_read_for_kaimei_to_run(v1):
    from titanic_kaimei.models import make_knn

    X, y = v1
    with pytest.raises(StructureError):  # why notebook 04 brings its own Shapley values or an endpoint
        extract_structure(make_knn().fit(X, y), feature_names=COLUMNS_V1)


def test_shapley_values_add_up_to_the_probability(v1):
    from titanic_kaimei.models import make_knn, shapley_values

    X, y = v1
    model = make_knn().fit(X, (y == "survived").astype(int))
    rows, background = X.iloc[:3], X.iloc[10:18]
    values, base = shapley_values(model, rows, background, permutations=2)
    assert values.shape == (3, X.shape[1])
    assert base + values.sum(axis=1) == pytest.approx(model.predict_proba(rows)[:, 1])


def test_accuracy_matches_the_notebooks(v1):
    X, y = v1
    table = scores({"forest": make_forest(), "logistic": make_logistic()}, X, y)
    # The notebooks quote these; a new scikit-learn may move them slightly, not by much.
    assert table.loc["forest", "5-fold CV accuracy"] == pytest.approx(0.827, abs=0.02)
    assert table.loc["logistic", "5-fold CV accuracy"] == pytest.approx(0.791, abs=0.02)


def test_v2_forest_is_as_accurate_and_executable():
    data = load(2)
    X, y = encode(data.passengers), data.survived
    table = scores({"forest": make_forest()}, X, y)
    assert table.loc["forest", "5-fold CV accuracy"] > 0.81
    forest = make_forest().fit(X, y)
    assert spec_is_executable(extract_structure(forest, feature_names=COLUMNS_V2).spec)
