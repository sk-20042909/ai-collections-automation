"""
Risk Segmentation Module
========================
Assigns risk tiers based on default-probability thresholds:
    Low Risk:       default_prob < 0.30
    Medium Risk:    0.30 <= default_prob < 0.60
    High Risk:      0.60 <= default_prob < 0.80
    Very High Risk: default_prob >= 0.80

A composite *priority_score* (0-100) is also computed to enable
fine-grained ranking within each tier.
"""

import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def _load_model_and_data():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    df = pd.read_csv(os.path.join(DATA_DIR, "borrowers_processed.csv"))
    return model, df


def _get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix (drop ID + target)."""
    import json
    cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(cols_path) as f:
        feature_cols = json.load(f)
    return df[feature_cols]


def assign_risk_tier(prob: float) -> str:
    if prob < 0.30:
        return "Low Risk"
    if prob < 0.60:
        return "Medium Risk"
    if prob < 0.80:
        return "High Risk"
    return "Very High Risk"


def compute_priority_score(row: pd.Series) -> float:
    """
    Weighted composite score (0-100).
    Higher = more urgent for collections outreach.
    """
    prob = row.get("default_probability", 0.5)
    credit_util = row.get("credit_utilization_index", 0.5)
    delinquency = row.get("delinquency_score", 0)
    max_delinq = max(delinquency, 1)  # avoid /0
    score = (
        prob * 40
        + min(credit_util, 1.0) * 25
        + min(delinquency / max_delinq, 1.0) * 25
        + np.random.uniform(0, 10)  # tiny jitter for tie-breaking
    )
    return round(float(np.clip(score, 0, 100)), 2)


def segment(df: pd.DataFrame | None = None):
    """
    Run segmentation.
    Returns a DataFrame with borrower_id, default_probability, risk_tier,
    and priority_score.
    """
    model, data = _load_model_and_data()
    if df is None:
        df = data

    X = _get_features(df)
    probs = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "borrower_id": df["borrower_id"].values,
        "default_probability": np.round(probs, 4),
    })
    result["repayment_probability"] = np.round(1.0 - result["default_probability"], 4)
    result["risk_tier"] = result["default_probability"].apply(assign_risk_tier)

    # Merge useful columns for priority calculation
    for col in ["credit_utilization_index", "delinquency_score"]:
        if col in df.columns:
            result[col] = df[col].values
    result["priority_score"] = result.apply(compute_priority_score, axis=1)

    # Drop helper columns
    result.drop(columns=["credit_utilization_index", "delinquency_score"],
                errors="ignore", inplace=True)

    tier_counts = result["risk_tier"].value_counts()
    print("[segmentation] Tier distribution:")
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count}")

    out_path = os.path.join(DATA_DIR, "risk_segments.csv")
    result.to_csv(out_path, index=False)
    print(f"[segmentation] Saved -> {out_path}")
    return result


if __name__ == "__main__":
    segment()
