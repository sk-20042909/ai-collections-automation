"""
Data Preprocessing Pipeline (UCI Credit Card Dataset)
======================================================
- Handles missing values
- Encodes categorical variables
- Scales numerical features
- Engineers five derived features:
    * debt_to_income_ratio    – avg bill amount / credit_limit
    * payment_behavior_score  – total payments / total positive bills (capped 0-1)
    * credit_utilization_index – avg bill / credit_limit (clipped 0-1)
    * delinquency_score       – weighted sum of PAY_x delay months
    * payment_consistency     – std-dev of monthly payment amounts (lower = more consistent)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

PAY_STATUS_COLS = ["pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"]
BILL_COLS = ["bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr"]
PAY_AMT_COLS = ["pay_amt_sep", "pay_amt_aug", "pay_amt_jul", "pay_amt_jun", "pay_amt_may", "pay_amt_apr"]

CATEGORICAL_COLS = ["gender", "education", "marital_status"]

NUMERIC_COLS = [
    "credit_limit", "age",
] + PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def load_raw(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(BASE_DIR, "data", "raw", "borrowers.csv")
    return pd.read_csv(path)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create five derived risk features."""
    avg_bill = df[BILL_COLS].mean(axis=1)

    # 1. debt_to_income_ratio
    df["debt_to_income_ratio"] = np.where(
        df["credit_limit"] > 0, avg_bill / df["credit_limit"], 0
    )

    # 2. payment_behavior_score
    total_paid = df[PAY_AMT_COLS].sum(axis=1)
    total_billed = df[BILL_COLS].clip(lower=0).sum(axis=1)
    df["payment_behavior_score"] = np.where(
        total_billed > 0, np.clip(total_paid / total_billed, 0, 1), 1.0
    )

    # 3. credit_utilization_index
    df["credit_utilization_index"] = np.clip(df["debt_to_income_ratio"], 0, 1)

    # 4. delinquency_score
    weights = np.array([6, 5, 4, 3, 2, 1], dtype=float)
    pay_vals = df[PAY_STATUS_COLS].clip(lower=0).values
    df["delinquency_score"] = (pay_vals * weights).sum(axis=1) / weights.sum()

    # 5. payment_consistency (lower std = more consistent)
    df["payment_consistency"] = df[PAY_AMT_COLS].std(axis=1)

    return df


def encode_and_scale(
    df: pd.DataFrame, fit: bool = True, scaler_path: str | None = None
) -> pd.DataFrame:
    # Preserve original categorical values for dashboard use
    cat_originals = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            cat_originals[col] = df[col].copy()

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Restore original categorical columns (used by dashboard, excluded from training)
    for col, values in cat_originals.items():
        df[col] = values.values

    engineered = [
        "debt_to_income_ratio", "payment_behavior_score",
        "credit_utilization_index", "delinquency_score", "payment_consistency",
    ]
    all_numeric = [c for c in NUMERIC_COLS + engineered if c in df.columns]

    if scaler_path is None:
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    if fit:
        scaler = StandardScaler()
        df[all_numeric] = scaler.fit_transform(df[all_numeric])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[preprocessing] Scaler saved -> {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[all_numeric] = scaler.transform(df[all_numeric])

    return df


def run_pipeline(raw_path: str | None = None, output_dir: str | None = None) -> str:
    df = load_raw(raw_path)
    df = handle_missing(df)
    df = engineer_features(df)
    df = encode_and_scale(df, fit=True)

    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "borrowers_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"[preprocessing] Saved processed data ({len(df)} rows) -> {out_path}")
    return out_path


if __name__ == "__main__":
    run_pipeline()
