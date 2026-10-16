# 🧪 AI Model Comparison Lab

Upload any CSV, choose a target column, and compare **Logistic Regression**, **Random Forest** and
**Gradient Boosting** side by side, with cross-validated hyperparameter search, a full set of
classification metrics, and a plain-language recommendation of which model to use and why.

Built with Python, pandas, scikit-learn, Plotly and Streamlit.

## Features

| Area | What it does |
|---|---|
| **Data profiling** | Row/column counts, duplicates, per-column type inference (numeric, categorical, boolean, identifier, free text, datetime, constant), missing-value counts and chart, summary statistics, distributions, data-quality warnings |
| **Preprocessing** | Median imputation plus missing-indicator flags for numeric columns; a dedicated "missing" category and one-hot encoding for categoricals; IDs, constants, free text and dates are dropped automatically. Everything runs **inside** the model pipeline, so validation folds never leak into imputation or scaling |
| **Models** | Logistic Regression, Random Forest, Histogram Gradient Boosting |
| **Tuning** | Randomized hyperparameter search with stratified k-fold CV (Quick / Balanced / Thorough budgets), refit on your chosen selection metric |
| **Metrics** | Accuracy, precision, recall, F1, ROC-AUC: CV mean ± std **and** hold-out test score. Confusion matrices (counts or row %), ROC curves, per-class report, per-fold scores |
| **Explainability** | Permutation importance on the hold-out set, measured on the original columns |
| **Recommendation** | Picks the model by CV score and applies the **one-standard-error rule** (prefer the simpler model when the difference is within noise). Reports head-to-head fold wins, checks the hold-out set agrees, and flags overfitting, high variance and class imbalance |

Binary and multiclass targets (2–20 classes) are supported. For multiclass, precision, recall and F1
are macro-averaged and ROC-AUC is one-vs-rest.

## Methodology

1. Rows with a missing target are removed. The data is split into a stratified training set and a
   hold-out test set (default 80/20).
2. For each model, `RandomizedSearchCV` explores its hyperparameter grid using **the same**
   `StratifiedKFold` splits, so fold scores can be compared pairwise across models.
3. The best configuration (by the selection metric) is refit on the full training set and evaluated
   once on the untouched hold-out set.
4. The recommendation ranks models by **mean CV score**, which is lower-variance than a single split.
   Any model within one standard error (`std / √k`) of the top scorer is treated as tied, and the
   simplest tied model wins (Logistic Regression < Random Forest < Gradient Boosting).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Three sample datasets are bundled in `data/` (churn with mixed types and missing values, breast
cancer, wine), so you can try the app without uploading anything. Regenerate them with
`python data/make_samples.py`.

## Tests

```bash
pip install pytest
pytest
```

The suite covers profiling, data preparation, binary and multiclass training, the recommendation,
and a headless end-to-end run of the Streamlit app.

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to <https://share.streamlit.io>, click **Create app**, and select the repository, branch and
   `app.py` as the entry point.
3. Click **Deploy**. `requirements.txt` is installed automatically.

Uploads are limited to 50 MB (`.streamlit/config.toml`), and datasets over 100,000 rows are sampled to keep
training within the free tier's resources.

## Project structure

```
app.py                  Streamlit UI
ml_lab/profiling.py     Column type inference, missing values, target validation
ml_lab/modeling.py      Preprocessing pipeline, model specs, CV search, evaluation, permutation importance
ml_lab/recommend.py     Model selection (one-SE rule) and written rationale
ml_lab/plots.py         Plotly figures
data/                   Sample datasets and generator script
tests/                  Unit and end-to-end tests
```

## Ideas for v2

- Regression targets (RMSE, MAE, R²)
- Decision-threshold tuning for imbalanced binary problems
- Probability calibration curves
- SHAP explanations for the selected model
- Export the trained pipeline (`joblib`) for reuse
