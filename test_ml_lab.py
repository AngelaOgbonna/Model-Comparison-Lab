import numpy as np
import pandas as pd
import pytest

from ml_lab.modeling import MODEL_SPECS, prepare_data, run_experiment
from ml_lab.profiling import candidate_targets, infer_column_kind, profile_dataframe, validate_target
from ml_lab.recommend import recommend


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    color = rng.choice(["red", "green", "blue"], n)
    y = np.where(x1 + (color == "red") + rng.normal(0, 0.5, n) > 0.5, "yes", "no")
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "x1": x1,
            "x2": rng.normal(size=n),
            "color": color,
            "const": 1,
            "label": y,
        }
    )
    df.loc[rng.random(n) < 0.1, "x1"] = np.nan
    df.loc[rng.random(n) < 0.1, "color"] = np.nan
    return df


def test_profile_detects_missing_and_column_kinds(mixed_df):
    p = profile_dataframe(mixed_df)
    cols = p.columns.set_index("column")
    assert cols.loc["x1", "missing"] == mixed_df["x1"].isna().sum() > 0
    assert cols.loc["id", "kind"] == "identifier"
    assert cols.loc["const", "kind"] == "constant"
    assert cols.loc["color", "kind"] == "categorical"
    assert p.total_missing_cells == mixed_df.isna().sum().sum()


def test_infer_kind_boolean_and_text():
    assert infer_column_kind(pd.Series(["Yes", "No", "Yes"])) == "boolean"
    assert infer_column_kind(pd.Series([f"word{i % 80}" for i in range(200)])) == "text"


def test_target_validation():
    df = pd.DataFrame({"cont": np.linspace(0, 1, 100), "cls": ["a", "b"] * 50, "one": ["a"] * 100})
    assert validate_target(df, "cls")[0]
    assert not validate_target(df, "cont")[0]
    assert not validate_target(df, "one")[0]
    assert candidate_targets(df)[0] == "cls"


def test_prepare_data_drops_unusable_columns_and_encodes_positive_class(mixed_df):
    data = prepare_data(mixed_df, "label", positive_class="yes")
    assert data.class_names == ["no", "yes"]
    assert set(data.dropped_features) == {"id", "const"}
    assert data.categorical_features == ["color"]
    assert (data.y == (mixed_df["label"] == "yes").to_numpy()).all()


def test_experiment_and_recommendation(mixed_df):
    data = prepare_data(mixed_df, "label", positive_class="yes")
    exp = run_experiment(data, list(MODEL_SPECS), primary_metric="roc_auc", cv_folds=3, search_budget="Quick")
    assert set(exp.models) == set(MODEL_SPECS)
    for r in exp.models.values():
        assert 0.5 < r.cv_mean["roc_auc"] <= 1.0
        assert r.confusion.sum() == exp.n_test
        assert len(r.cv_fold_scores["roc_auc"]) == 3
    rec = recommend(exp, imbalance_ratio=1.0)
    assert rec.model in MODEL_SPECS
    assert "Recommended model" in rec.to_markdown()


def test_multiclass_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(240, 3))
    y = np.array(["a", "b", "c"])[np.argmax(X + rng.normal(0, 0.3, X.shape), axis=1)]
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"]).assign(target=y)
    data = prepare_data(df, "target")
    exp = run_experiment(data, ["Logistic Regression"], cv_folds=3, search_budget="Quick")
    r = exp.models["Logistic Regression"]
    assert r.confusion.shape == (3, 3)
    assert r.test_metrics["roc_auc"] > 0.8
