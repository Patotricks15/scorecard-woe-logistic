import pytest

from scorecard_woe.pipeline import CATEGORICAL, FEATURES, NUMERIC, PROTECTED, evaluate, fit_scorecard, load_split


@pytest.fixture(scope="module")
def split():
    return load_split()


def test_split_is_stratified_and_keeps_protected_out_of_features(split):
    assert len(split.X_train) == 700 and len(split.X_test) == 300
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.02
    assert set(FEATURES).isdisjoint(PROTECTED)


def test_protected_attribute_is_not_used_by_model(split):
    sc, _ = fit_scorecard(split, C=1.0)
    table = sc.points_table()
    assert "Sex" not in set(table["feature"])


def test_scorecard_beats_random_on_holdout(split):
    sc, _ = fit_scorecard(split, C=1.0)
    m = evaluate(sc, split)
    assert m["test"]["auc"] > 0.6
    assert m["test"]["gini"] == pytest.approx(2 * m["test"]["auc"] - 1)


def test_chosen_C_is_on_grid(split):
    _, C = fit_scorecard(split)
    assert 0.01 <= C <= 10


def test_numeric_and_categorical_partition(split):
    assert set(NUMERIC).isdisjoint(CATEGORICAL)
