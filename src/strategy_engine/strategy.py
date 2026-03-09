"""
Strategy Engine – Collections & Recovery
=========================================
Maps risk tiers to recommended collection actions using a blend of
rule-based logic and ML-driven priority scores.

Four strategy tiers:
    Low Risk       -> Automated SMS / email reminders
    Medium Risk    -> Call-center outreach + payment plan
    High Risk      -> Escalated recovery team + restructured payment
    Very High Risk -> Legal review / settlement offer
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

STRATEGY_MAP: dict[str, dict] = {
    "Low Risk": {
        "action": "Automated SMS / email reminder",
        "channel": "SMS + Email",
        "urgency": "Low",
        "follow_up_days": 14,
        "description": "Send periodic payment reminders via automated channels.",
    },
    "Medium Risk": {
        "action": "Call-center outreach + payment plan offer",
        "channel": "Phone + Email",
        "urgency": "Medium",
        "follow_up_days": 7,
        "description": (
            "Proactive call-center contact with a customised payment plan "
            "offer based on the borrower's financial profile."
        ),
    },
    "High Risk": {
        "action": "Escalated recovery team + restructured payment",
        "channel": "Phone + In-Person",
        "urgency": "High",
        "follow_up_days": 3,
        "description": (
            "Assign a senior recovery agent. Offer loan restructuring "
            "or extended repayment terms."
        ),
    },
    "Very High Risk": {
        "action": "Legal review / settlement offer",
        "channel": "Legal + Phone",
        "urgency": "Critical",
        "follow_up_days": 1,
        "description": (
            "Initiate legal review process. Present a settlement offer "
            "or negotiate a reduced lump-sum payment."
        ),
    },
}


def assign_strategy(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich *segments_df* (output of segmenter) with strategy columns.
    """
    strategy_rows = []
    for _, row in segments_df.iterrows():
        tier = row["risk_tier"]
        info = STRATEGY_MAP.get(tier, STRATEGY_MAP["Medium Risk"])
        strategy_rows.append({
            "borrower_id": row["borrower_id"],
            "risk_tier": tier,
            "default_probability": row["default_probability"],
            "priority_score": row["priority_score"],
            "recommended_action": info["action"],
            "channel": info["channel"],
            "urgency": info["urgency"],
            "follow_up_days": info["follow_up_days"],
            "description": info["description"],
        })
    df = pd.DataFrame(strategy_rows)

    out = os.path.join(DATA_DIR, "collection_strategies.csv")
    df.to_csv(out, index=False)
    print(f"[strategy_engine] {len(df)} strategies assigned -> {out}")
    return df


def run_strategy() -> pd.DataFrame:
    segments = pd.read_csv(os.path.join(DATA_DIR, "risk_segments.csv"))
    return assign_strategy(segments)


if __name__ == "__main__":
    run_strategy()
