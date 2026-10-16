"""Regenerate the bundled sample CSVs: `python data/make_samples.py`."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

OUT = Path(__file__).parent


def customer_churn(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Synthetic telecom churn data with mixed types, missing values and an ID column."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No internet"], n, p=[0.35, 0.45, 0.20])
    monthly = np.round(20 + (internet == "Fiber optic") * 45 + (internet == "DSL") * 25 + rng.normal(0, 12, n), 2)
    monthly = np.clip(monthly, 18, None)
    support_calls = rng.poisson(1.2, n)
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    age = np.clip(rng.normal(45, 15, n).round(), 18, 90)
    region = rng.choice(["North", "South", "East", "West"], n)

    logit = (
        -1.2
        - 0.045 * tenure
        + 1.3 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic")
        + 0.012 * (monthly - 60)
        + 0.35 * support_calls
        + 0.3 * (paperless == "Yes")
        + 0.6 * (support_calls >= 4) * (contract == "Month-to-month")  # interaction trees can find
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": np.arange(100001, 100001 + n),
            "age": age,
            "region": region,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "monthly_charges": monthly,
            "total_charges": np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2),
            "support_calls": support_calls,
            "paperless_billing": paperless,
            "churn": np.where(churn, "Yes", "No"),
        }
    )
    # Inject missingness: some at random, some informative.
    for col, frac in {"age": 0.12, "total_charges": 0.03, "internet_service": 0.05, "monthly_charges": 0.02}.items():
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def main() -> None:
    customer_churn().to_csv(OUT / "customer_churn.csv", index=False)

    bc = load_breast_cancer(as_frame=True)
    frame = bc.frame.copy()
    frame["diagnosis"] = np.where(frame.pop("target") == 1, "benign", "malignant")
    frame.to_csv(OUT / "breast_cancer.csv", index=False)

    wine = load_wine(as_frame=True)
    frame = wine.frame.copy()
    frame["cultivar"] = frame.pop("target").map({0: "class_0", 1: "class_1", 2: "class_2"})
    frame.to_csv(OUT / "wine.csv", index=False)


if __name__ == "__main__":
    main()
