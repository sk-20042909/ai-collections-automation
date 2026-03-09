"""
Data Preprocessing Pipeline
============================
- Handles missing values (median / mode imputation)
- Encodes categorical variables (one-hot encoding)
- Scales numerical features (StandardScaler)
- Engineers new features:
    * debt_to_income_ratio
    * payment_behavior_score
    * delinquency_index
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

CATEGORICAL_COLS = [
    "employment_status",
    "preferred_contact_channel",
    "region",
    "repayment_status",
]

NUMERIC_COLS = [
    "age",
    "monthly_income",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "loan_tenure_months",
    "emi_amount",
    "days_past_due",
    "number_of_missed_payments",
    "last_payment_amount",
    "contact_response_rate",
    "prior_collection_attempts",
    "outstanding_balance",
    "customer_tenure",
]


def load_raw(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw", "borrowers.csv"
        )
    return pd.read_csv(path)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values – median for numeric, mode for categorical."""
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived risk features."""
    df["debt_to_income_ratio"] = np.where(
        df["monthly_income"] > 0,
        df["outstanding_balance"] / (df["monthly_income"] * 12),
        0,
    )
    df["payment_behavior_score"] = (
        0.4 * df["contact_response_rate"]
        + 0.3 * (1 - df["number_of_missed_payments"] / 12)
        + 0.3 * np.clip(df["last_payment_amount"] / (df["emi_amount"] + 1), 0, 1)
    )
    df["delinquency_index"] = (
        df["days_past_due"] / 360 * 0.5
        + df["number_of_missed_payments"] / 12 * 0.3
        + df["prior_collection_attempts"] / 10 * 0.2
    )
    return df


def encode_and_scale(
    df: pd.DataFrame, fit: bool = True, scaler_path: str | None = None
) -> pd.DataFrame:
    """One-hot encode categoricals and standard-scale numerics."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    new_numeric = ["debt_to_income_ratio", "payment_behavior_score", "delinquency_index"]
    all_numeric = [c for c in NUMERIC_COLS + new_numeric if c in df.columns]

    if scaler_path is None:
        scaler_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "scaler.pkl"
        )

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
    """Execute the full preprocessing pipeline and save processed CSV."""
    df = load_raw(raw_path)
    df = handle_missing(df)
    df = engineer_features(df)
    df = encode_and_scale(df, fit=True)

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "processed"
        )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "borrowers_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"[preprocessing] Saved processed data -> {out_path}")
    return out_path


if __name__ == "__main__":
    run_pipeline()
