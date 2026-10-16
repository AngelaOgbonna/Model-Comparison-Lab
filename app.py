"""AI Model Comparison Lab — upload a CSV, pick a target, train and compare classifiers."""

from __future__ import annotations

import html
import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import auth
import plots
import styles
from modeling import (
    METRIC_LABELS,
    METRICS,
    MODEL_SPECS,
    SEARCH_BUDGETS,
    feature_importance,
    prepare_data,
    run_experiment,
)
from profiling import candidate_targets, class_balance, imbalance_ratio, profile_dataframe, validate_target
from recommend import recommend

DATA_DIR = Path(__file__).parent
SAMPLES = {
    "Customer churn": ("customer_churn.csv", "Binary · mixed types · missing values"),
    "Breast cancer diagnosis": ("breast_cancer.csv", "Binary · 30 numeric features"),
    "Wine cultivar": ("wine.csv", "3 classes · numeric"),
}
MAX_ROWS = 100_000
# Widget values that should survive moving between pages.
PERSISTENT_KEYS = ["source", "sample", "target", "positive", "exclude", "models", "metric", "folds", "test_size", "budget"]

st.set_page_config(page_title="Model Comparison Lab", page_icon=":material/science:", layout="wide")
styles.inject()

for _k in PERSISTENT_KEYS:  # re-assigning keeps widget state alive when its page isn't rendered
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]
# Defaults live in session state (not in the widgets), so they don't clash with the persistence above.
for _k, _v in {"source": "Sample", "folds": 5, "test_size": 0.2, "budget": "Balanced", "models": list(MODEL_SPECS)}.items():
    st.session_state.setdefault(_k, _v)


# --------------------------------------------------------------------------- helpers


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame):
    return profile_dataframe(df)


def chart(fig) -> None:
    fig.update_layout(font_family="Inter, sans-serif", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def md_to_html(text: str) -> str:
    """Escape, then turn the **bold** and `code` markers used in generated text into HTML."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", out)


def set_dataset(df: pd.DataFrame, name: str) -> None:
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0).reset_index(drop=True)
    if st.session_state.get("dataset_name") != name:
        for key in ("experiment", "prepared", "importance", "target", "positive", "exclude", "metric"):
            st.session_state.pop(key, None)
    st.session_state.df = df
    st.session_state.dataset_name = name


def dataset_label() -> str:
    name = st.session_state.dataset_name
    return next((n for n, (f, _) in SAMPLES.items() if f == name), name)


def ensure_dataset() -> None:
    if "df" not in st.session_state:
        file, _ = SAMPLES["Customer churn"]
        set_dataset(load_csv((DATA_DIR / file).read_bytes()), file)


def go(page_key: str, label: str, icon: str, primary: bool = False) -> None:
    if st.button(label, icon=icon, type="primary" if primary else "secondary", key=f"go_{page_key}_{label}"):
        st.switch_page(PAGES[page_key])


# --------------------------------------------------------------------------- pages


def overview_page() -> None:
    ensure_dataset()
    user = auth.current_user()
    styles.hero(
        "Welcome" if user["name"] == "Guest" else f"Welcome, {user['name'].split()[0]}",
        "Find the best model for your data.",
        "Upload any CSV, choose what to predict, and watch Logistic Regression, Random Forest and "
        "Gradient Boosting compete under cross-validation. You get a recommendation with the reasoning.",
        ["Automatic profiling", "Hyperparameter tuning", "ROC-AUC · F1 · Precision · Recall", "Explainable choice"],
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        styles.step_card("STEP 01", "Load your data", "Upload a CSV or pick a sample. We profile every column and flag missing values.")
    with c2:
        styles.step_card("STEP 02", "Train the models", "Pick a target and settings. Each model is tuned with stratified cross-validation.")
    with c3:
        styles.step_card("STEP 03", "Compare & decide", "Explore metrics, ROC curves and confusion matrices, then read the recommendation.")

    st.write("")
    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        go("data", "Start with data", ":material/arrow_forward:", primary=True)
    with b2:
        go("train", "Jump to training", ":material/model_training:")

    st.write("")
    df = st.session_state.df
    exp = st.session_state.get("experiment")
    winner = recommend(exp, st.session_state.imbalance).model if exp else "—"
    styles.kpis(
        [
            ("folder_open", "Active dataset", dataset_label(), f"{len(df):,} rows × {df.shape[1]} columns"),
            ("track_changes", "Target", st.session_state.get("trained_target", "Not chosen"), "What the models predict"),
            ("smart_toy", "Models", "3", "LR · RF · Gradient Boosting"),
            ("workspace_premium", "Last winner", winner, "From your latest run" if exp else "Train to see a result"),
        ]
    )


def data_page() -> None:
    ensure_dataset()
    styles.page_header("Data", "Load a dataset and check its quality before training.")

    with st.container(border=True):
        left, right = st.columns([1, 2], gap="large")
        with left:
            source = st.segmented_control("Data source", ["Sample", "Upload CSV"], key="source") or "Sample"
        with right:
            if source == "Upload CSV":
                upload = st.file_uploader("CSV file", type=["csv", "txt"], label_visibility="collapsed")
                if upload is not None:
                    try:
                        set_dataset(load_csv(upload.getvalue()), upload.name)
                    except Exception as exc:  # noqa: BLE001 — show any parse error
                        st.error(f"Could not read the file: {exc}")
            else:
                names = list(SAMPLES)
                if st.session_state.get("sample") not in names:
                    st.session_state.sample = next(
                        (n for n, (f, _) in SAMPLES.items() if f == st.session_state.dataset_name), names[0]
                    )
                choice = st.selectbox(
                    "Sample dataset",
                    names,
                    format_func=lambda n: f"{n}  ·  {SAMPLES[n][1]}",
                    key="sample",
                )
                file, _ = SAMPLES[choice]
                if file != st.session_state.dataset_name:
                    set_dataset(load_csv((DATA_DIR / file).read_bytes()), file)

    df = st.session_state.df
    profile = cached_profile(df)
    styles.kpis(
        [
            ("table_rows", "Rows", f"{profile.n_rows:,}", st.session_state.dataset_name),
            ("view_column", "Columns", str(profile.n_cols), f"{(profile.columns['kind'] == 'numeric').sum()} numeric"),
            ("grid_off", "Missing cells", f"{profile.missing_pct:.1f}%", f"{profile.total_missing_cells:,} empty values"),
            ("content_copy", "Duplicate rows", str(profile.n_duplicate_rows), "Exact repeated rows"),
        ]
    )
    for w in profile.warnings:
        styles.note(md_to_html(w))

    t_cols, t_missing, t_preview, t_dist = st.tabs(["Columns", "Missing values", "Preview", "Distributions"])
    with t_cols:
        st.dataframe(
            profile.columns,
            width="stretch",
            hide_index=True,
            column_config={
                "missing_%": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f%%"),
                **{c: st.column_config.NumberColumn(format="%.3g") for c in ("mean", "std", "min", "max")},
            },
        )
    with t_missing:
        if profile.total_missing_cells:
            chart(plots.missing_values_chart(profile.columns))
            st.caption(
                "Gaps are filled inside each model pipeline (median + a missing-flag for numbers, a “missing” "
                "category for text), so the validation data never leaks into training."
            )
        else:
            st.success("No missing values detected.")
    with t_preview:
        st.dataframe(df.head(100), width="stretch")
    with t_dist:
        numeric_cols = profile.columns.query("kind == 'numeric'")["column"].tolist()
        if numeric_cols:
            col = st.selectbox("Numeric column", numeric_cols)
            chart(plots.numeric_distribution(df[col], col))
        else:
            st.info("No numeric columns.")

    st.write("")
    go("train", "Continue to training", ":material/arrow_forward:", primary=True)


def train_page() -> None:
    ensure_dataset()
    df = st.session_state.df
    styles.page_header("Train", f"Configure the experiment for {st.session_state.dataset_name}.")

    left, right = st.columns([1.05, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("#### What should the models predict?")
        targets = candidate_targets(df)
        if st.session_state.get("target") not in targets:
            st.session_state.target = targets[0]
        target = st.selectbox("Target column", targets, key="target", help="A categorical column with 2–20 classes.")
        ok, msg = validate_target(df, target)
        if not ok:
            st.error(msg)
            return
        y_full = df[target].dropna().astype(str)
        classes = sorted(y_full.unique())
        positive = None
        if len(classes) == 2:
            if st.session_state.get("positive") not in classes:
                st.session_state.positive = y_full.value_counts().idxmin()
            positive = st.selectbox(
                "Positive class", classes, key="positive", help="Precision, recall and F1 describe this class."
            )
        options = [c for c in df.columns if c != target]
        st.session_state.exclude = [c for c in st.session_state.get("exclude", []) if c in options]
        exclude = st.multiselect(
            "Leave out columns",
            options,
            key="exclude",
            help="Drop anything that leaks the answer. IDs, constants, free text and dates are dropped automatically.",
        )
        models = st.pills("Models", list(MODEL_SPECS), selection_mode="multi", key="models")

    with right:
        chart(plots.class_balance_chart(class_balance(y_full), target))

    with st.expander("Advanced settings", icon=":material/tune:"):
        a, b, c, d = st.columns(4)
        if "metric" not in st.session_state:
            st.session_state.metric = "f1" if imbalance_ratio(y_full) >= 1.5 else "accuracy"
        metric = a.selectbox("Choose the winner by", METRICS, format_func=METRIC_LABELS.get, key="metric")
        folds = b.slider("Cross-validation folds", 3, 10, key="folds")
        test_size = c.slider("Test set size", 0.1, 0.4, step=0.05, key="test_size")
        budget = d.select_slider(
            "Tuning effort",
            list(SEARCH_BUDGETS),
            key="budget",
            help=", ".join(f"{k}: {v} settings tried per model" for k, v in SEARCH_BUDGETS.items()),
        )

    if y_full.value_counts().min() * (1 - test_size) < folds:
        st.error("The smallest class is too small for this many folds — lower the number of folds.")
        return

    st.write("")
    if st.button("Train & compare models", type="primary", icon=":material/rocket_launch:", disabled=not models):
        data = prepare_data(df, target, exclude=exclude, positive_class=positive)
        if not data.numeric_features and not data.categorical_features:
            st.error("No usable feature columns remain after exclusions.")
            return
        bar = st.progress(0.0, text="Starting…")
        try:
            exp = run_experiment(
                data,
                models,
                primary_metric=metric,
                cv_folds=folds,
                test_size=test_size,
                search_budget=budget,
                progress=lambda frac, text: bar.progress(frac, text=text),
            )
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.exception(exc)
            return
        st.session_state.update(
            experiment=exp, prepared=data, imbalance=imbalance_ratio(y_full), trained_target=target, importance={}
        )
        st.switch_page(PAGES["results"])


def results_page() -> None:
    exp = st.session_state.get("experiment")
    if exp is None:
        styles.page_header("Results", "Nothing here yet.")
        styles.note("Train the models first and the comparison will appear here.", kind="info")
        go("train", "Go to training", ":material/model_training:", primary=True)
        return

    data = st.session_state.prepared
    rec = recommend(exp, st.session_state.imbalance)
    metric = exp.primary_metric
    label = METRIC_LABELS[metric]
    styles.page_header(
        "Results",
        f"{st.session_state.dataset_name} · predicting “{st.session_state.trained_target}” · "
        f"{exp.cv_folds}-fold CV · {exp.n_train:,} train / {exp.n_test:,} test rows",
    )

    chosen = exp.models[rec.model]
    styles.recommendation(
        rec.model,
        f"Cross-validated {label} of <b>{chosen.cv_mean[metric]:.3f}</b> "
        f"(± {chosen.cv_std[metric]:.3f}) and <b>{chosen.test_metrics[metric]:.3f}</b> on unseen test data.",
        [md_to_html(r) for r in rec.reasons[2:5]],
    )

    board = exp.leaderboard()
    styles.model_cards(
        [
            {
                "name": name,
                "color": plots.MODEL_COLORS[name],
                "score": f"{exp.models[name].cv_mean[metric]:.3f}",
                "score_label": f"CV {label} · ±{exp.models[name].cv_std[metric]:.3f}",
                "mini": [
                    ("Acc", f"{exp.models[name].test_metrics['accuracy']:.2f}"),
                    ("Prec", f"{exp.models[name].test_metrics['precision']:.2f}"),
                    ("Recall", f"{exp.models[name].test_metrics['recall']:.2f}"),
                    ("AUC", f"{exp.models[name].test_metrics['roc_auc']:.2f}"),
                ],
                "winner": name == rec.model,
            }
            for name in board["model"]
        ]
    )

    with st.expander("Full reasoning & caveats", icon=":material/psychology:"):
        st.markdown(rec.to_markdown())
        st.write(f"**Numeric features ({len(data.numeric_features)}):** {', '.join(data.numeric_features) or '—'}")
        st.write(f"**Categorical features ({len(data.categorical_features)}):** {', '.join(data.categorical_features) or '—'}")
        if data.dropped_features:
            st.write("**Dropped:** " + ", ".join(f"{k} ({v})" for k, v in data.dropped_features.items()))
        for w in exp.warnings:
            st.caption(f"Note: {w}")

    t_board, t_cv, t_roc, t_cm, t_imp, t_params = st.tabs(
        ["Leaderboard", "Cross-validation", "ROC curves", "Confusion matrices", "Feature importance", "Settings found"]
    )

    with t_board:
        a, b = st.columns(2, gap="medium")
        with a:
            chart(plots.metrics_comparison(exp, "cv"))
        with b:
            chart(plots.metrics_comparison(exp, "test"))
        view = pd.DataFrame({"Model": board["model"]})
        for m in METRICS:
            view[f"CV {METRIC_LABELS[m]}"] = [
                f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(board[f"cv_{m}"], board[f"cv_{m}_std"], strict=True)
            ]
        for m in METRICS:
            view[f"Test {METRIC_LABELS[m]}"] = board[f"test_{m}"].round(3)
        view["Train time (s)"] = board["fit_seconds"].round(1)
        st.dataframe(view, width="stretch", hide_index=True)
        st.caption(
            f"CV = mean ± std over {exp.cv_folds} folds of the training data. Test = the untouched hold-out set."
            + ("" if exp.is_binary else " Precision/recall/F1 are macro-averaged; ROC-AUC is one-vs-rest.")
        )
        st.download_button(
            "Download results (CSV)",
            board.to_csv(index=False).encode(),
            file_name=f"leaderboard_{Path(st.session_state.dataset_name).stem}.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    with t_cv:
        metric_view = st.segmented_control(
            "Metric", METRICS, default=metric, format_func=METRIC_LABELS.get, key="cv_metric"
        ) or metric
        chart(plots.cv_folds_chart(exp, metric_view))
        folds = pd.DataFrame(
            {name: r.cv_fold_scores[metric_view] for name, r in exp.models.items()},
            index=[f"Fold {i + 1}" for i in range(exp.cv_folds)],
        )
        folds.loc["Mean"] = folds.mean()
        st.dataframe(folds.style.format("{:.3f}"), width="stretch")
        st.caption("Every model is scored on the same folds, so each row is a fair head-to-head.")

    with t_roc:
        chart(plots.roc_chart(exp))

    with t_cm:
        normalize = st.toggle("Show percentages (recall per class)", value=True)
        cols = st.columns(len(exp.models), gap="medium")
        for col, r in zip(cols, exp.models.values(), strict=True):
            with col:
                chart(plots.confusion_chart(r.confusion, exp.class_names, r.name, normalize))
        pick = st.selectbox("Per-class report for", list(exp.models), index=list(exp.models).index(rec.model))
        st.dataframe(pd.DataFrame(exp.models[pick].class_report).T.style.format("{:.3f}"), width="stretch")

    with t_imp:
        pick = st.selectbox("Model", list(exp.models), index=list(exp.models).index(rec.model), key="imp_model")
        cache = st.session_state.setdefault("importance", {})
        if pick not in cache:
            with st.spinner("Measuring which columns matter…"):
                cache[pick] = feature_importance(exp.models[pick], exp.X_test, exp.y_test, exp.is_binary, metric)
        chart(plots.importance_chart(cache[pick], metric, pick))
        st.caption(
            f"Each column is shuffled in turn; the bar shows how much test {label} drops. Bigger drop = more important."
        )

    with t_params:
        cols = st.columns(len(exp.models), gap="medium")
        for col, r in zip(cols, exp.models.values(), strict=True):
            with col, st.container(border=True):
                st.markdown(f"**{r.name}**")
                st.caption(MODEL_SPECS[r.name].description)
                st.json({k: v if v is None or isinstance(v, (int, float, str)) else str(v) for k, v in r.best_params.items()})
                st.caption(f"Best of {r.n_candidates} settings tried")


# --------------------------------------------------------------------------- navigation

PAGES = {
    "overview": st.Page(overview_page, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    "data": st.Page(data_page, title="Data", icon=":material/table_chart:", url_path="data"),
    "train": st.Page(train_page, title="Train", icon=":material/model_training:", url_path="train"),
    "results": st.Page(results_page, title="Results", icon=":material/insights:", url_path="results"),
}

user = auth.current_user()
if user is None:
    st.navigation([st.Page(auth.sign_in_page, title="Sign in", icon=":material/login:")], position="hidden").run()
else:
    with st.sidebar:
        styles.sidebar_profile(user["name"], user["email"])
        st.html('<div class="side-label">Dataset</div>')
        if "df" in st.session_state:
            st.caption(f"{st.session_state.dataset_name} · {len(st.session_state.df):,} rows")
        else:
            st.caption("Customer churn sample (default)")
        st.html('<div class="side-label">Appearance</div>')
        st.caption("Switch dark / light (black & white): **⋮ menu → Settings → Theme**.")
        st.write("")
        if st.button("Sign out", icon=":material/logout:", width="stretch"):
            auth.sign_out()
            st.rerun()
    st.navigation(list(PAGES.values()), position="top").run()
