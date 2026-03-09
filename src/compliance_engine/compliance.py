"""
Compliance Engine
==================
Applies regulatory and ethical rules before any contact is made.

Rules
-----
1. Do not contact outside allowed hours (08:00 – 20:00).
2. Limit contact attempts to a maximum of 5 per borrower.
3. Protect vulnerable customers (age > 60 AND low income).
4. Avoid aggressive communication (High-risk borrowers with
   prior_collection_attempts >= 8 must not receive threatening language).

Returns ``compliance_status`` (PASS / FAIL) and ``violation_reason``.
"""

import os
import datetime
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

MAX_CONTACT_ATTEMPTS = 5
ALLOWED_HOUR_START = 8
ALLOWED_HOUR_END = 20
VULNERABLE_AGE = 60
VULNERABLE_INCOME = 15000
AGGRESSIVE_THRESHOLD_ATTEMPTS = 8


def check_compliance(
    actions_path: str | None = None,
    raw_path: str | None = None,
    current_hour: int | None = None,
) -> pd.DataFrame:
    """
    Merge collection actions with the raw borrower data and flag violations.
    """
    if actions_path is None:
        actions_path = os.path.join(
            BASE_DIR, "data", "processed", "collection_actions.csv"
        )
    if raw_path is None:
        raw_path = os.path.join(BASE_DIR, "data", "raw", "borrowers.csv")

    actions = pd.read_csv(actions_path)
    raw = pd.read_csv(raw_path)[
        ["borrower_id", "age", "monthly_income", "prior_collection_attempts"]
    ]
    df = actions.merge(raw, on="borrower_id", how="left")

    if current_hour is None:
        current_hour = datetime.datetime.now().hour

    violations = []
    for _, row in df.iterrows():
        reasons = []

        # Rule 1 – contact hours
        if current_hour < ALLOWED_HOUR_START or current_hour >= ALLOWED_HOUR_END:
            reasons.append("Outside allowed contact hours (08:00-20:00)")

        # Rule 2 – max attempts
        if row.get("prior_collection_attempts", 0) >= MAX_CONTACT_ATTEMPTS:
            reasons.append(
                f"Max contact attempts exceeded ({int(row['prior_collection_attempts'])} >= {MAX_CONTACT_ATTEMPTS})"
            )

        # Rule 3 – vulnerable customer
        if row.get("age", 0) >= VULNERABLE_AGE and row.get("monthly_income", 999999) <= VULNERABLE_INCOME:
            reasons.append("Vulnerable customer (age >= 60 & low income)")

        # Rule 4 – aggressive communication
        if (
            row.get("risk_segment") == "High Risk"
            and row.get("prior_collection_attempts", 0) >= AGGRESSIVE_THRESHOLD_ATTEMPTS
        ):
            reasons.append("Aggressive communication risk – excessive prior attempts")

        violations.append("; ".join(reasons) if reasons else "")

    df["violation_reason"] = violations
    df["compliance_status"] = df["violation_reason"].apply(
        lambda v: "FAIL" if v else "PASS"
    )
    return df[
        ["borrower_id", "risk_segment", "recommended_action", "compliance_status", "violation_reason"]
    ]


def save_compliance(output_dir: str | None = None) -> str:
    df = check_compliance()
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "compliance_flags.csv")
    df.to_csv(path, index=False)
    flagged = (df["compliance_status"] == "FAIL").sum()
    print(f"[compliance_engine] {flagged}/{len(df)} flagged  -> {path}")
    return path


if __name__ == "__main__":
    save_compliance()
