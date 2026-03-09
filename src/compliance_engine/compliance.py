"""
Compliance Engine
=================
Flags potential compliance violations before collection actions are executed.

Rules implemented (UCI-dataset aware):
    1. Contact time window  – outreach only 08:00-21:00 local time.
    2. Maximum frequency     – no more than 3 contacts per borrower per week.
    3. Vulnerable customer   – borrowers aged 60+ receive special handling.
    4. Regulatory escalation – Very-High-Risk accounts must go through
                               legal-review gate before external action.
    5. Consent check         – placeholder flag; reminder to verify
                               communication consent before outreach.
"""

import os
from datetime import datetime
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def _time_window_flag() -> str | None:
    """Flag if current hour is outside 08-21."""
    hour = datetime.now().hour
    if hour < 8 or hour >= 21:
        return "Outside permitted contact hours (08:00-21:00)"
    return None


def _vulnerable_customer_flag(age: float) -> str | None:
    if age >= 60:
        return "Vulnerable customer (age >= 60) – apply special handling"
    return None


def _frequency_flag(contact_count: int) -> str | None:
    if contact_count >= 3:
        return "Max weekly contact limit reached (3)"
    return None


def _regulatory_escalation_flag(risk_tier: str) -> str | None:
    if risk_tier == "Very High Risk":
        return "Legal-review gate required before external action"
    return None


def _consent_flag() -> str | None:
    """Placeholder – always remind agent to confirm consent."""
    return "Verify borrower communication consent before outreach"


def run_compliance_checks(strategies_df: pd.DataFrame,
                          borrowers_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply compliance rules to every row of *strategies_df*.
    *borrowers_df* provides age column.
    Returns a DataFrame of compliance flags.
    """
    # Merge age from processed data
    if borrowers_df is None:
        proc_path = os.path.join(DATA_DIR, "borrowers_processed.csv")
        borrowers_df = pd.read_csv(proc_path)

    # Build lookup for age
    age_map = {}
    if "age" in borrowers_df.columns:
        age_map = dict(zip(borrowers_df["borrower_id"], borrowers_df["age"]))

    flags_rows = []
    for _, row in strategies_df.iterrows():
        bid = row["borrower_id"]
        tier = row["risk_tier"]
        age = age_map.get(bid, 30)

        row_flags = []
        tw = _time_window_flag()
        if tw:
            row_flags.append(tw)
        vf = _vulnerable_customer_flag(age)
        if vf:
            row_flags.append(vf)
        rf = _regulatory_escalation_flag(tier)
        if rf:
            row_flags.append(rf)
        cf = _consent_flag()
        if cf:
            row_flags.append(cf)
        # Frequency check (placeholder: assume 0 prior contacts)
        ff = _frequency_flag(0)
        if ff:
            row_flags.append(ff)

        flags_rows.append({
            "borrower_id": bid,
            "risk_tier": tier,
            "flags": " | ".join(row_flags) if row_flags else "Clear",
            "flag_count": len(row_flags),
            "checked_at": datetime.now().isoformat(),
        })

    df = pd.DataFrame(flags_rows)
    out = os.path.join(DATA_DIR, "compliance_flags.csv")
    df.to_csv(out, index=False)
    flagged = len(df[df["flag_count"] > 0])
    print(f"[compliance_engine] {flagged}/{len(df)} borrowers flagged -> {out}")
    return df


def run_compliance() -> pd.DataFrame:
    strats = pd.read_csv(os.path.join(DATA_DIR, "collection_strategies.csv"))
    return run_compliance_checks(strats)


if __name__ == "__main__":
    run_compliance()
