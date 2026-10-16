"""AI Model Comparison Lab — upload a CSV, pick a target, train and compare classifiers."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from ml_lab import plots
from ml_lab.modeling import (
    METRIC_LABELS,
    METRICS,
    MODEL_SPECS,
    SEARCH_BUDGETS,
    feature_importance,
    prepare_data,
    run_experiment,
)
from ml_lab.profiling import (
    candidate_targets,
    class_balance,
    imbalance_ratio,
    profile_dataframe,
    validate_target,
)
from ml_lab.recommend import recommend

DATA_DIR = Path(__file__).parent / "data"
SAMPLES = {
    "Customer churn (binary, mixed types, missing values)": "customer_churn.csv",
    "Breast cancer diagnosis (binary, numeric)": "breast_cancer.csv",
    "Wine cultivar (3 classes)": "wine.csv",
}
MAX_UPLOAD_ROWS = 100_000

st.set_page_config(page_title="AI Model Comparison Lab", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- cached helpers


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame):
    return profile_dataframe(df)


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# --------------------------------------------------------------------------- sidebar: data source

st.sidebar.title("🧪 Model Comparison Lab")
source = st.sidebar.radio("Data source", ["Upload a CSV", "Use a sample dataset"], index=1)

df: pd.DataFrame | None = None
dataset_name = ""
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader("CSV file", type=["csv", "txt"])
    if upload is not None:
        try:
            df = load_csv(upload.getvalue())
            dataset_name = upload.name
        except Exception as exc:  # noqa: BLE001 — surface any parse error to the user
            st.sidebar.error(f"Could not read the file: {exc}")
else:
    sample = st.sidebar.selectbox("Sample", list(SAMPLES))
    df = load_csv((DATA_DIR / SAMPLES[sample]).read_bytes())
    dataset_name = SAMPLES[sample]

st.title("AI Model Comparison Lab")
st.caption(
    "Upload a dataset, choose what to predict, and compare Logistic Regression, Random Forest and "
    "Gradient Boosting with cross-validated hyperparameter search."
)

if df is None:
    st.info("⬅️ Upload a CSV file (or pick a sample dataset) to get started.")
    st.stop()

if len(df) > MAX_UPLOAD_ROWS:
    st.warning(f"The dataset has {len(df):,} rows; a random sample of {MAX_UPLOAD_ROWS:,} is used to keep training fast.")
    df = df.sample(MAX_UPLOAD_ROWS, random_state=0).reset_index(drop=True)

# Reset results whenever the dataset changes.
if st.session_state.get("dataset_name") != dataset_name:
    st.session_state.dataset_name = dataset_name
    st.session_state.pop("experiment", None)

profile = cached_profile(df)

# --------------------------------------------------------------------------- 1. profile

st.header("1 · Data profile")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile.n_rows:,}")
c2.metric("Columns", profile.n_cols)
c3.metric("Missing cells", f"{profile.total_missing_cells:,}", f"{profile.missing_pct:.1f}% of all cells", delta_color="off")
c4.metric("Duplicate rows", profile.n_duplicate_rows)

for w in profile.warnings:
    st.warning(w, icon="⚠️")

tab_cols, tab_missing, tab_preview, tab_dist = st.tabs(["Columns", "Missing values", "Preview", "Distributions"])
with tab_cols:
    st.dataframe(
        profile.columns,
        width="stretch",
        hide_index=True,
        column_config={
            "missing_%": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f%%"),
            "mean": st.column_config.NumberColumn(format="%.3g"),
            "std": st.column_config.NumberColumn(format="%.3g"),
            "min": st.column_config.NumberColumn(format="%.3g"),
            "max": st.column_config.NumberColumn(format="%.3g"),
        },
    )
with tab_missing:
    if profile.total_missing_cells:
        chart(plots.missing_values_chart(profile.columns))
        st.caption(
            "Missing values are imputed inside each model pipeline (median for numeric columns, plus a "
            "missing-indicator flag; a dedicated “missing” category for categorical columns), so imputation "
            "never sees the validation folds."
        )
    else:
        st.success("No missing values detected.")
with tab_preview:
    st.dataframe(df.head(50), width="stretch")
with tab_dist:
    numeric_cols = profile.columns.query("kind == 'numeric'")["column"].tolist()
    if numeric_cols:
        col = st.selectbox("Numeric column", numeric_cols)
        chart(plots.numeric_distribution(df[col], col))
    else:
        st.write("No numeric columns.")

# --------------------------------------------------------------------------- 2. configure

st.header("2 · Configure the experiment")
targets = candidate_targets(df)
left, right = st.columns(2)
with left:
    target = st.selectbox("Target variable (what to predict)", targets, help="Must be categorical: 2–20 classes.")
    ok, msg = validate_target(df, target)
    if not ok:
        st.error(msg)
        st.stop()

    y_full = df[target].dropna().astype(str)
    classes = sorted(y_full.unique())
    positive = None
    if len(classes) == 2:
        minority = y_full.value_counts().idxmin()
        positive = st.selectbox(
            "Positive class",
            classes,
            index=classes.index(minority),
            help="Precision, recall and F1 are reported for this class. Defaults to the rarer class.",
        )
    feature_options = [c for c in df.columns if c != target]
    exclude = st.multiselect(
        "Exclude columns from features",
        feature_options,
        help="Drop anything that leaks the answer or would not be available at prediction time. "
        "IDs, constants, free text and dates are dropped automatically.",
    )
    models = st.multiselect("Models", list(MODEL_SPECS), default=list(MODEL_SPECS))

with right:
    chart(plots.class_balance_chart(class_balance(y_full), target))

with st.expander("Training settings", expanded=True):
    s1, s2, s3, s4 = st.columns(4)
    default_metric = "f1" if imbalance_ratio(y_full) >= 1.5 else "accuracy"
    primary_metric = s1.selectbox(
        "Selection metric",
        METRICS,
        index=METRICS.index(default_metric),
        format_func=METRIC_LABELS.get,
        help="Hyperparameters and the recommended model are chosen by this metric.",
    )
    cv_folds = s2.slider("Cross-validation folds", 3, 10, 5)
    test_size = s3.slider("Hold-out test size", 0.1, 0.4, 0.2, 0.05)
    budget = s4.select_slider(
        "Hyperparameter search",
        list(SEARCH_BUDGETS),
        value="Balanced",
        help=", ".join(f"{k}: {v} candidates per model" for k, v in SEARCH_BUDGETS.items()),
    )

min_class = y_full.value_counts().min()
if min_class * (1 - test_size) < cv_folds:
    st.error(f"The smallest class has only {min_class} rows — reduce the number of CV folds.")
    st.stop()

if st.button("🚀 Train and compare", type="primary", disabled=not models):
    data = prepare_data(df, target, exclude=exclude, positive_class=positive)
    if not data.numeric_features and not data.categorical_features:
        st.error("No usable feature columns remain after exclusions.")
        st.stop()
    bar = st.progress(0.0, text="Starting…")
    try:
        exp = run_experiment(
            data,
            models,
            primary_metric=primary_metric,
            cv_folds=cv_folds,
            test_size=test_size,
            search_budget=budget,
            progress=lambda frac, text: bar.progress(frac, text=text),
        )
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        st.exception(exc)
        st.stop()
    bar.empty()
    st.session_state.experiment = exp
    st.session_state.prepared = data
    st.session_state.imbalance = imbalance_ratio(y_full)
    st.session_state.target = target
    st.session_state.pop("importance", None)

# --------------------------------------------------------------------------- 3. results

exp = st.session_state.get("experiment")
if exp is None:
    st.stop()
data = st.session_state.prepared
if st.session_state.target != target:
    st.info("Settings changed since the last run — press **Train and compare** to refresh the results.")

st.header("3 · Results")
rec = recommend(exp, st.session_state.imbalance)

with st.container(border=True):
    st.markdown(rec.to_markdown())

with st.expander("Features used"):
    st.write(f"**Numeric ({len(data.numeric_features)}):** {', '.join(data.numeric_features) or '—'}")
    st.write(f"**Categorical ({len(data.categorical_features)}):** {', '.join(data.categorical_features) or '—'}")
    if data.dropped_features:
        st.write("**Dropped:** " + ", ".join(f"{k} ({v})" for k, v in data.dropped_features.items()))
    if data.rows_dropped_missing_target:
        st.write(f"{data.rows_dropped_missing_target} rows with a missing target were removed.")
    for w in exp.warnings:
        st.caption(f"⚠️ {w}")

t_board, t_cv, t_roc, t_cm, t_imp, t_params = st.tabs(
    ["Leaderboard", "Cross-validation", "ROC curves", "Confusion matrices", "Feature importance", "Hyperparameters"]
)

with t_board:
    board = exp.leaderboard()
    view = pd.DataFrame({"Model": board["model"]})
    for m in METRICS:
        view[f"CV {METRIC_LABELS[m]}"] = [
            f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(board[f"cv_{m}"], board[f"cv_{m}_std"], strict=True)
        ]
    for m in METRICS:
        view[f"Test {METRIC_LABELS[m]}"] = board[f"test_{m}"].round(3)
    view["Train time (s)"] = board["fit_seconds"].round(1)
    view.insert(1, "", ["⭐ recommended" if n == rec.model else "" for n in board["model"]])
    st.dataframe(view, width="stretch", hide_index=True)
    st.caption(
        f"Sorted by CV {METRIC_LABELS[exp.primary_metric]}. CV columns: mean ± std over {exp.cv_folds} folds "
        f"of the training set ({exp.n_train} rows). Test columns: untouched hold-out set ({exp.n_test} rows)."
        + ("" if exp.is_binary else " Precision/recall/F1 are macro-averaged; ROC-AUC is one-vs-rest.")
    )
    a, b = st.columns(2)
    with a:
        chart(plots.metrics_comparison(exp, "cv"))
    with b:
        chart(plots.metrics_comparison(exp, "test"))
    st.download_button(
        "Download leaderboard (CSV)",
        board.to_csv(index=False).encode(),
        file_name=f"leaderboard_{Path(dataset_name).stem}.csv",
        mime="text/csv",
    )

with t_cv:
    metric_view = st.selectbox(
        "Metric", METRICS, index=METRICS.index(exp.primary_metric), format_func=METRIC_LABELS.get, key="cv_metric"
    )
    chart(plots.cv_folds_chart(exp, metric_view))
    folds = pd.DataFrame(
        {name: r.cv_fold_scores[metric_view] for name, r in exp.models.items()},
        index=[f"Fold {i + 1}" for i in range(exp.cv_folds)],
    )
    folds.loc["Mean"] = folds.mean()
    folds.loc["Std"] = folds.iloc[:-1].std(ddof=0)
    st.dataframe(folds.style.format("{:.3f}"), width="stretch")
    st.caption(
        "Every model is evaluated on the same folds, so rows can be compared directly. "
        "Std uses the population formula to match scikit-learn's reported values."
    )

with t_roc:
    chart(plots.roc_chart(exp))

with t_cm:
    normalize = st.toggle("Show row percentages (recall per class)", value=True)
    cols = st.columns(len(exp.models))
    for col, r in zip(cols, exp.models.values(), strict=True):
        with col:
            chart(plots.confusion_chart(r.confusion, exp.class_names, r.name, normalize))
    with st.expander("Per-class report"):
        pick = st.selectbox("Model", list(exp.models), index=list(exp.models).index(rec.model), key="report_model")
        report = pd.DataFrame(exp.models[pick].class_report).T
        st.dataframe(report.style.format("{:.3f}"), width="stretch")

with t_imp:
    pick = st.selectbox("Model", list(exp.models), index=list(exp.models).index(rec.model), key="imp_model")
    cache = st.session_state.setdefault("importance", {})
    if pick not in cache:
        with st.spinner("Computing permutation importance on the hold-out set…"):
            cache[pick] = feature_importance(exp.models[pick], exp.X_test, exp.y_test, exp.is_binary, exp.primary_metric)
    chart(plots.importance_chart(cache[pick], exp.primary_metric, pick))
    st.caption(
        "Permutation importance shuffles one original column at a time and measures how much the hold-out "
        f"{METRIC_LABELS[exp.primary_metric]} drops. It works identically for every model type, and is "
        "computed on raw columns, so one-hot encoded categories are grouped back together."
    )

with t_params:
    for r in exp.models.values():
        st.subheader(r.name)
        st.caption(MODEL_SPECS[r.name].description)
        st.json({k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in r.best_params.items()})
        grid = MODEL_SPECS[r.name].param_grid
        st.caption(
            f"Randomized search over {r.n_candidates} of the "
            f"{pd.Series([len(v) for v in grid.values()]).prod()} combinations in: "
            + "; ".join(f"`{k}` ∈ {v}" for k, v in grid.items())
        )
