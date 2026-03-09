"""
Risk Segmentation Module
=========================
Uses the trained model to predict repayment probability for every borrower,
then assigns a risk segment and a priority score.

Segments
--------
- Low Risk    : probability > 0.75
- Medium Risk : 0.45 ≤ probability ≤ 0.75
- High Risk   : probability < 0.45
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _load_model():
    return joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))


def _load_columns():
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        return json.load(f)


def _align_columns(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """Ensure df has exactly the columns the model expects."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]


def segment_borrowers(processed_path: str | None = None) -> pd.DataFrame:
    """Predict repayment probability and assign risk segments."""
    if processed_path is None:
        processed_path = os.path.join(
            BASE_DIR, "data", "processed", "borrowers_processed.csv"
        )
    df = pd.read_csv(processed_path)

    model = _load_model()
    feature_cols = _load_columns()

    # Keep borrower_id for output
    ids = df["borrower_id"] if "borrower_id" in df.columns else pd.Series(range(len(df)))

    X = _align_columns(df.drop(columns=["recovered", "borrower_id"], errors="ignore"), feature_cols)
    probs = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "borrower_id": ids,
        "repayment_probability": np.round(probs, 4),
    })

    result["risk_segment"] = pd.cut(
        result["repayment_probability"],
        bins=[-0.01, 0.45, 0.75, 1.01],
        labels=["High Risk", "Medium Risk", "Low Risk"],
    )

    # Priority score: higher = more urgent (inverse of probability)
    result["priority_score"] = np.round(1 - result["repayment_probability"], 4)

    return result


def save_segments(output_dir: str | None = None) -> str:
    """Run segmentation and save to CSV."""
    df = segment_borrowers()
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "risk_segments.csv")
    df.to_csv(path, index=False)
    print(f"[segmentation] Saved {len(df)} segments -> {path}")
    return path


if __name__ == "__main__":
    save_segments()
