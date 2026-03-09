"""
Synthetic Borrower Dataset Generator
=====================================
Generates a realistic dataset of 5,000+ borrowers with demographic,
financial, behavioural and loan attributes.  The binary target variable
``recovered`` (1 = debt recovered, 0 = not recovered) is produced from a
probabilistic model so downstream ML is meaningful.

Feature Descriptions
--------------------
- borrower_id            : Unique identifier for each borrower
- age                    : Borrower age (21–65)
- employment_status      : Employed / Self-Employed / Unemployed / Retired
- monthly_income         : Monthly income in local currency
- credit_score           : Credit bureau score (300–850)
- loan_amount            : Original sanctioned loan amount
- interest_rate          : Annual interest rate (%)
- loan_tenure_months     : Total loan duration in months
- emi_amount             : Equated monthly instalment
- days_past_due          : Days since last missed payment
- number_of_missed_payments : Count of missed EMIs
- last_payment_amount    : Amount of the most recent payment
- contact_response_rate  : Fraction of contact attempts borrower responded to
- prior_collection_attempts : How many times collections contacted borrower
- preferred_contact_channel : SMS / Email / Phone / WhatsApp
- outstanding_balance    : Current remaining balance
- region                 : Geographic region
- customer_tenure        : Months since account was opened
- repayment_status       : Current / Late / Defaulted
- recovered              : TARGET – 1 if debt was recovered, else 0
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_BORROWERS = 5000

REGIONS = ["North", "South", "East", "West", "Central"]
EMPLOYMENT_STATUSES = ["Employed", "Self-Employed", "Unemployed", "Retired"]
CONTACT_CHANNELS = ["SMS", "Email", "Phone", "WhatsApp"]


def _dpd_distribution(n_values: int) -> np.ndarray:
    """Skewed probability distribution for days_past_due (most near 0)."""
    x = np.arange(n_values, dtype=float)
    p = np.exp(-x / 60)
    return p / p.sum()


def generate_dataset(n: int = N_BORROWERS, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with *n* synthetic borrower records."""
    rng = np.random.RandomState(seed)

    data = {
        "borrower_id": [f"BRW-{i+1:05d}" for i in range(n)],
        "age": rng.randint(21, 66, size=n),
        "employment_status": rng.choice(
            EMPLOYMENT_STATUSES, size=n, p=[0.50, 0.25, 0.15, 0.10]
        ),
        "monthly_income": np.round(
            rng.normal(45000, 15000, size=n).clip(8000, 150000), 2
        ),
        "credit_score": rng.randint(300, 851, size=n),
        "loan_amount": np.round(rng.uniform(10000, 500000, size=n), 2),
        "interest_rate": np.round(rng.uniform(7.5, 24.0, size=n), 2),
        "loan_tenure_months": rng.choice([6, 12, 24, 36, 48, 60], size=n),
        "days_past_due": rng.choice(
            np.arange(0, 361), size=n, p=_dpd_distribution(361)
        ),
        "number_of_missed_payments": rng.randint(0, 13, size=n),
        "contact_response_rate": np.round(rng.beta(2, 5, size=n), 2),
        "prior_collection_attempts": rng.randint(0, 11, size=n),
        "preferred_contact_channel": rng.choice(CONTACT_CHANNELS, size=n),
        "region": rng.choice(REGIONS, size=n),
        "customer_tenure": rng.randint(1, 121, size=n),
    }

    df = pd.DataFrame(data)

    # EMI calculation (standard reducing-balance formula)
    r = df["interest_rate"] / 1200
    t = df["loan_tenure_months"]
    df["emi_amount"] = np.round(
        df["loan_amount"] * r * ((1 + r) ** t) / (((1 + r) ** t) - 1), 2
    )
    df["last_payment_amount"] = np.round(
        df["emi_amount"] * rng.uniform(0.0, 1.2, size=n), 2
    )
    df["outstanding_balance"] = np.round(
        df["loan_amount"] * rng.uniform(0.1, 1.0, size=n), 2
    )

    # Repayment status from days_past_due
    conditions = [
        df["days_past_due"] == 0,
        df["days_past_due"] <= 90,
        df["days_past_due"] > 90,
    ]
    df["repayment_status"] = np.select(
        conditions, ["Current", "Late", "Defaulted"], default="Late"
    )

    # TARGET: recovered (probabilistic based on key features)
    score = (
        0.25 * (df["credit_score"] - 300) / 550
        + 0.20 * (df["monthly_income"] - 8000) / 142000
        + 0.15 * (1 - df["days_past_due"] / 360)
        + 0.15 * df["contact_response_rate"]
        + 0.10 * (1 - df["number_of_missed_payments"] / 12)
        + 0.10 * (df["customer_tenure"] / 120)
        + 0.05 * (1 - df["prior_collection_attempts"] / 10)
    )
    prob = 1 / (1 + np.exp(-5 * (score - 0.45)))
    df["recovered"] = (rng.rand(n) < prob).astype(int)

    return df


def save_dataset(output_dir: str | None = None) -> str:
    """Generate and persist the dataset.  Returns the CSV path."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw"
        )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "borrowers.csv")
    df = generate_dataset()
    df.to_csv(path, index=False)
    print(f"[dataset_generator] Saved {len(df)} records -> {path}")
    return path


if __name__ == "__main__":
    save_dataset()
