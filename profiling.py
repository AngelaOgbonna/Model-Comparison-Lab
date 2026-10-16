"""Automatic data profiling: column types, missing values and data-quality warnings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Columns with more missing values than this are flagged as unreliable.
HIGH_MISSING_THRESHOLD = 0.40
# A categorical column with more distinct values than this is treated as free text / IDs.
MAX_CATEGORICAL_CARDINALITY = 50
# A target with more classes than this is not a sensible classification target.
MAX_TARGET_CLASSES = 20

BOOL_TOKENS = {"0", "1", "0.0", "1.0", "True", "False", "true", "false", "yes", "no", "Yes", "No", "Y", "N"}


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    total_missing_cells: int
    columns: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def missing_pct(self) -> float:
        cells = self.n_rows * self.n_cols
        return 100 * self.total_missing_cells / cells if cells else 0.0


def infer_column_kind(series: pd.Series) -> str:
    """Classify a column as numeric, categorical, boolean, datetime, identifier, text or constant."""
    non_null = series.dropna()
    n_unique = non_null.nunique()
    n = len(non_null)

    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series) or (n_unique == 2 and set(map(str, non_null.unique())) <= BOOL_TOKENS):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # Integer columns where every value is unique and monotonic are almost always row IDs.
        if pd.api.types.is_integer_dtype(non_null) and n_unique == n and n > 20 and (
            non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing
        ):
            return "identifier"
        return "numeric"
    if n_unique == n and n > 20:
        return "identifier"
    if n_unique > MAX_CATEGORICAL_CARDINALITY:
        return "text"
    return "categorical"


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    rows = []
    warnings: list[str] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_frac = missing / n if n else 0.0
        kind = infer_column_kind(s)
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "kind": kind,
            "missing": missing,
            "missing_%": round(100 * missing_frac, 2),
            "unique": int(s.nunique(dropna=True)),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "top_value": None,
        }
        if kind == "numeric":
            row.update(mean=s.mean(), std=s.std(), min=s.min(), max=s.max())
        elif s.notna().any():
            row["top_value"] = str(s.mode(dropna=True).iloc[0])
        rows.append(row)

        if missing_frac > HIGH_MISSING_THRESHOLD:
            warnings.append(
                f"**{col}** is {100 * missing_frac:.0f}% missing — consider excluding it."
            )
        if kind == "constant":
            warnings.append(f"**{col}** has a single value and carries no information; it will be dropped.")
        elif kind == "identifier":
            warnings.append(f"**{col}** looks like a row identifier; it will be excluded from features.")
        elif kind == "text":
            warnings.append(
                f"**{col}** has {row['unique']} distinct text values (high cardinality); it will be excluded."
            )
        elif kind == "datetime":
            warnings.append(f"**{col}** is a datetime and will be excluded (engineer features from it manually).")

    dupes = int(df.duplicated().sum())
    if dupes:
        warnings.append(f"{dupes} duplicate rows found — they may inflate evaluation scores.")

    return DatasetProfile(
        n_rows=n,
        n_cols=df.shape[1],
        n_duplicate_rows=dupes,
        total_missing_cells=int(df.isna().sum().sum()),
        columns=pd.DataFrame(rows),
        warnings=warnings,
    )


def candidate_targets(df: pd.DataFrame) -> list[str]:
    """Columns that make sensible classification targets, best guesses first."""
    good, rest = [], []
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if 2 <= n_unique <= MAX_TARGET_CLASSES:
            good.append(col)
        else:
            rest.append(col)
    # Datasets conventionally put the label last, so prefer the right-most suitable column.
    return list(reversed(good)) + rest


def validate_target(df: pd.DataFrame, target: str) -> tuple[bool, str]:
    """Check that the target is usable for classification. Returns (ok, message)."""
    y = df[target].dropna()
    n_classes = y.nunique()
    if n_classes < 2:
        return False, f"`{target}` has fewer than two classes, so there is nothing to predict."
    if n_classes > MAX_TARGET_CLASSES:
        msg = f"`{target}` has {n_classes} distinct values. This lab handles classification targets "
        msg += f"(at most {MAX_TARGET_CLASSES} classes)"
        if pd.api.types.is_numeric_dtype(y):
            msg += " — it looks continuous, which would need a regression model."
        return False, msg + "."
    counts = y.value_counts()
    if counts.min() < 5:
        return False, (
            f"Class `{counts.idxmin()}` has only {counts.min()} rows; every class needs at least 5 "
            "for stratified cross-validation."
        )
    return True, ""


def class_balance(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=True)
    return pd.DataFrame(
        {"class": counts.index.astype(str), "count": counts.values, "share_%": (100 * counts / counts.sum()).round(2)}
    )


def imbalance_ratio(y: pd.Series) -> float:
    counts = y.value_counts(dropna=True)
    return float(counts.max() / counts.min())
