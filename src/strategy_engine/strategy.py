"""
Collection Strategy Engine
===========================
Rule-based engine that maps risk segments to recommended collection actions.

+---------------+-----------------------------+-----------------+
| Risk Segment  | Recommended Action          | Urgency Level   |
+---------------+-----------------------------+-----------------+
| Low Risk      | SMS reminder                | Low             |
| Medium Risk   | Call-center follow-up       | Medium          |
| High Risk     | Field collection escalation | High            |
+---------------+-----------------------------+-----------------+
"""

import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

STRATEGY_MAP = {
    "Low Risk": {
        "recommended_action": "SMS reminder",
        "urgency_level": "Low",
    },
    "Medium Risk": {
        "recommended_action": "Call-center follow-up",
        "urgency_level": "Medium",
    },
    "High Risk": {
        "recommended_action": "Field collection escalation",
        "urgency_level": "High",
    },
}


def recommend_actions(segments_path: str | None = None) -> pd.DataFrame:
    """Attach recommended action and urgency to each borrower."""
    if segments_path is None:
        segments_path = os.path.join(
            BASE_DIR, "data", "processed", "risk_segments.csv"
        )
    df = pd.read_csv(segments_path)

    df["recommended_action"] = df["risk_segment"].map(
        lambda s: STRATEGY_MAP.get(s, {}).get("recommended_action", "Review manually")
    )
    df["urgency_level"] = df["risk_segment"].map(
        lambda s: STRATEGY_MAP.get(s, {}).get("urgency_level", "Unknown")
    )
    # Priority rank: 1 = most urgent
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df["priority_rank"] = df.index + 1

    return df


def save_actions(output_dir: str | None = None) -> str:
    df = recommend_actions()
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "collection_actions.csv")
    df.to_csv(path, index=False)
    print(f"[strategy_engine] Saved {len(df)} actions -> {path}")
    return path


if __name__ == "__main__":
    save_actions()
