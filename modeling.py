"""Preprocessing, model definitions, hyperparameter search and evaluation."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

from profiling import infer_column_kind

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}
# Number of random hyperparameter combinations tried per model.
SEARCH_BUDGETS = {"Quick": 4, "Balanced": 12, "Thorough": 30}


# --------------------------------------------------------------------------- data preparation


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    class_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_features: dict[str, str]
    rows_dropped_missing_target: int

    @property
    def is_binary(self) -> bool:
        return len(self.class_names) == 2


def prepare_data(
    df: pd.DataFrame,
    target: str,
    exclude: list[str] | None = None,
    positive_class: str | None = None,
) -> PreparedData:
    """Split a raw frame into features and an integer-encoded target.

    For binary problems the ``positive_class`` is encoded as 1, so precision/recall/F1 refer to it.
    """
    exclude = set(exclude or [])
    target_series = df[target]
    keep = target_series.notna()
    data = df.loc[keep].reset_index(drop=True)
    y_raw = data[target].astype(str)

    classes = sorted(y_raw.unique())
    if len(classes) == 2 and positive_class is not None and str(positive_class) in classes:
        classes = [c for c in classes if c != str(positive_class)] + [str(positive_class)]
    y = y_raw.map({c: i for i, c in enumerate(classes)}).to_numpy()

    numeric, categorical, dropped = [], [], {}
    for col in data.columns:
        if col == target:
            continue
        if col in exclude:
            dropped[col] = "excluded by user"
            continue
        kind = infer_column_kind(data[col])
        if kind in {"constant", "identifier", "text", "datetime"}:
            dropped[col] = kind
        elif kind in {"numeric", "boolean"} and pd.api.types.is_numeric_dtype(data[col]):
            numeric.append(col)
        else:
            categorical.append(col)

    X = data[numeric + categorical].copy()
    for col in numeric:
        X[col] = X[col].astype(float)
    for col in categorical:
        # Normalise to object dtype holding strings + NaN, so imputing and encoding behave the same
        # for ints-as-categories, booleans and pandas string dtypes.
        X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)

    return PreparedData(
        X=X,
        y=y,
        class_names=classes,
        numeric_features=numeric,
        categorical_features=categorical,
        dropped_features=dropped,
        rows_dropped_missing_target=int((~keep).sum()),
    )


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones.

    Lives inside each model pipeline so imputation statistics are learned on training folds only.
    """
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], object]
    param_grid: dict[str, list]
    complexity: int  # lower = simpler / more interpretable; used to break near-ties
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "Logistic Regression": ModelSpec(
        name="Logistic Regression",
        factory=lambda seed: LogisticRegression(max_iter=3000, random_state=seed),
        param_grid={
            "C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        complexity=1,
        description="Linear model; fast, well calibrated and directly interpretable via coefficients.",
    ),
    "Random Forest": ModelSpec(
        name="Random Forest",
        factory=lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [None, 6, 12, 20],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced"],
        },
        complexity=2,
        description="Bagged decision trees; captures non-linearities and interactions, robust to outliers.",
    ),
    "Gradient Boosting": ModelSpec(
        name="Gradient Boosting",
        factory=lambda seed: HistGradientBoostingClassifier(random_state=seed),
        param_grid={
            "learning_rate": [0.03, 0.1, 0.2],
            "max_iter": [100, 300],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [0.0, 1.0],
        },
        complexity=3,
        description="Sequentially boosted trees (histogram-based); usually the strongest on tabular data.",
    ),
}


def scoring_for(is_binary: bool) -> dict[str, object]:
    average = "binary" if is_binary else "macro"
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average=average, zero_division=0),
        "recall": make_scorer(recall_score, average=average, zero_division=0),
        "f1": make_scorer(f1_score, average=average, zero_division=0),
        "roc_auc": "roc_auc" if is_binary else "roc_auc_ovr",
    }


# --------------------------------------------------------------------------- training & evaluation


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    best_params: dict
    n_candidates: int
    cv_mean: dict[str, float]
    cv_std: dict[str, float]
    cv_fold_scores: dict[str, list[float]]
    train_mean: dict[str, float]
    test_metrics: dict[str, float]
    confusion: np.ndarray
    roc: tuple[np.ndarray, np.ndarray]
    class_report: dict
    fit_seconds: float


@dataclass
class ExperimentResult:
    models: dict[str, ModelResult]
    primary_metric: str
    cv_folds: int
    test_size: float
    class_names: list[str]
    n_train: int
    n_test: int
    X_test: pd.DataFrame
    y_test: np.ndarray
    warnings: list[str] = field(default_factory=list)

    @property
    def is_binary(self) -> bool:
        return len(self.class_names) == 2

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for r in self.models.values():
            row = {"model": r.name}
            for m in METRICS:
                row[f"cv_{m}"] = r.cv_mean[m]
                row[f"cv_{m}_std"] = r.cv_std[m]
            for m in METRICS:
                row[f"test_{m}"] = r.test_metrics[m]
            row["fit_seconds"] = r.fit_seconds
            rows.append(row)
        board = pd.DataFrame(rows)
        return board.sort_values(f"cv_{self.primary_metric}", ascending=False).reset_index(drop=True)


def _test_metrics(y_true, y_pred, proba, is_binary: bool) -> dict[str, float]:
    average = "binary" if is_binary else "macro"
    if is_binary:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", labels=np.arange(proba.shape[1]))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "roc_auc": auc,
    }


def _roc(y_true, proba, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Binary ROC curve, or the micro-averaged one-vs-rest curve for multiclass."""
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
    else:
        y_bin = LabelBinarizer().fit(np.arange(n_classes)).transform(y_true)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
    return fpr, tpr


def run_experiment(
    data: PreparedData,
    model_names: list[str],
    primary_metric: str = "f1",
    cv_folds: int = 5,
    test_size: float = 0.2,
    search_budget: str = "Balanced",
    seed: int = 42,
    progress: Callable[[float, str], None] | None = None,
) -> ExperimentResult:
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=test_size, stratify=data.y, random_state=seed
    )
    # A shared splitter means every model is scored on identical folds -> paired comparisons are valid.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = scoring_for(data.is_binary)
    n_classes = len(data.class_names)
    budget = SEARCH_BUDGETS[search_budget]

    results: dict[str, ModelResult] = {}
    run_warnings: list[str] = []
    for i, name in enumerate(model_names):
        spec = MODEL_SPECS[name]
        if progress:
            progress(i / len(model_names), f"Tuning {name} ({cv_folds}-fold CV)…")

        pipe = Pipeline(
            [
                ("prep", build_preprocessor(data.numeric_features, data.categorical_features)),
                ("model", spec.factory(seed)),
            ]
        )
        grid = {f"model__{k}": v for k, v in spec.param_grid.items()}
        n_iter = min(budget, len(ParameterGrid(grid)))
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=n_iter,
            scoring=scoring,
            refit=primary_metric,
            cv=cv,
            return_train_score=True,
            random_state=seed,
            n_jobs=-1,
            error_score="raise",
        )
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            run_warnings.append(f"{name}: some candidates did not fully converge.")

        cvr = search.cv_results_
        idx = search.best_index_
        model = search.best_estimator_
        proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        results[name] = ModelResult(
            name=name,
            estimator=model,
            best_params={k.removeprefix("model__"): v for k, v in search.best_params_.items()},
            n_candidates=n_iter,
            cv_mean={m: float(cvr[f"mean_test_{m}"][idx]) for m in METRICS},
            cv_std={m: float(cvr[f"std_test_{m}"][idx]) for m in METRICS},
            cv_fold_scores={
                m: [float(cvr[f"split{k}_test_{m}"][idx]) for k in range(cv_folds)] for m in METRICS
            },
            train_mean={m: float(cvr[f"mean_train_{m}"][idx]) for m in METRICS},
            test_metrics=_test_metrics(y_test, y_pred, proba, data.is_binary),
            confusion=confusion_matrix(y_test, y_pred, labels=np.arange(n_classes)),
            roc=_roc(y_test, proba, n_classes),
            class_report=classification_report(
                y_test,
                y_pred,
                labels=np.arange(n_classes),
                target_names=data.class_names,
                output_dict=True,
                zero_division=0,
            ),
            fit_seconds=elapsed,
        )

    if progress:
        progress(1.0, "Done")

    return ExperimentResult(
        models=results,
        primary_metric=primary_metric,
        cv_folds=cv_folds,
        test_size=test_size,
        class_names=data.class_names,
        n_train=len(X_train),
        n_test=len(X_test),
        X_test=X_test,
        y_test=y_test,
        warnings=run_warnings,
    )


def feature_importance(
    result: ModelResult, X_test: pd.DataFrame, y_test: np.ndarray, is_binary: bool, metric: str, seed: int = 42
) -> pd.DataFrame:
    """Permutation importance on the held-out set, measured on the original (pre-encoding) columns."""
    if len(X_test) > 2000:
        sample = X_test.sample(2000, random_state=seed)
        y_sample = y_test[X_test.index.get_indexer(sample.index)]
    else:
        sample, y_sample = X_test, y_test
    imp = permutation_importance(
        result.estimator,
        sample,
        y_sample,
        scoring=scoring_for(is_binary)[metric],
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    )
    return (
        pd.DataFrame({"feature": sample.columns, "importance": imp.importances_mean, "std": imp.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
