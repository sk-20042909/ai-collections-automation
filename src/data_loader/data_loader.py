"""
Data Loader – UCI Credit Card Default Dataset
===============================================
Loads the UCI Default of Credit Card Clients dataset, cleans column names,
handles missing / invalid values, and saves a clean CSV for downstream use.

Dataset source:
    https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains 30,000 records and 23 features plus a binary target:
    ``default`` (1 = default next month, 0 = no default).

Column Mapping (original → cleaned)
------------------------------------
ID               → borrower_id
LIMIT_BAL        → credit_limit
SEX              → gender            (1=Male, 2=Female)
EDUCATION        → education         (1=Grad, 2=University, 3=HighSchool, 4+=Other)
MARRIAGE         → marital_status    (1=Married, 2=Single, 3=Other)
AGE              → age
PAY_0 .. PAY_6   → pay_sep .. pay_apr (repayment status –1=on-time, 1–9=months delay)
BILL_AMT1..6     → bill_sep .. bill_apr
PAY_AMT1..6      → pay_amt_sep .. pay_amt_apr
default payment next month → default  (TARGET)
"""

import os
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Friendlier column names
RENAME_MAP = {
    "ID": "borrower_id",
    "LIMIT_BAL": "credit_limit",
    "SEX": "gender",
    "EDUCATION": "education",
    "MARRIAGE": "marital_status",
    "AGE": "age",
    "PAY_0": "pay_sep",
    "PAY_2": "pay_aug",
    "PAY_3": "pay_jul",
    "PAY_4": "pay_jun",
    "PAY_5": "pay_may",
    "PAY_6": "pay_apr",
    "BILL_AMT1": "bill_sep",
    "BILL_AMT2": "bill_aug",
    "BILL_AMT3": "bill_jul",
    "BILL_AMT4": "bill_jun",
    "BILL_AMT5": "bill_may",
    "BILL_AMT6": "bill_apr",
    "PAY_AMT1": "pay_amt_sep",
    "PAY_AMT2": "pay_amt_aug",
    "PAY_AMT3": "pay_amt_jul",
    "PAY_AMT4": "pay_amt_jun",
    "PAY_AMT5": "pay_amt_may",
    "PAY_AMT6": "pay_amt_apr",
    "default payment next month": "default",
}


def adapt_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Schema adapter: normalise column names to the system schema.

    Handles both already-renamed frames and raw UCI frames so the rest
    of the pipeline always sees a consistent schema.
    """
    # Lowercase all column names first
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Apply the rename map (case-insensitive lookup)
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    df = df.rename(columns=lower_map)

    # Generate borrower_id if missing
    if "borrower_id" not in df.columns:
        df.insert(0, "borrower_id",
                  [f"BRW-{i+1:05d}" for i in range(len(df))])
    return df


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load, clean, and return the UCI credit-card default dataset."""
    if path is None:
        path = os.path.join(RAW_DIR, "uci_credit.xls")

    df = pd.read_excel(path, header=1)
    df = df.rename(columns=RENAME_MAP)

    # Prefix borrower_id for readability
    df["borrower_id"] = df["borrower_id"].apply(lambda x: f"BRW-{int(x):05d}")

    # Ensure all columns are lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Clean invalid education & marital_status codes
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marital_status"] = df["marital_status"].replace({0: 3})

    # Drop rows with any remaining nulls (dataset is almost clean)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def save_clean_csv(output_dir: str | None = None) -> str:
    """Load the raw XLS and persist as a clean CSV."""
    df = load_dataset()
    if output_dir is None:
        output_dir = RAW_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "borrowers.csv")
    df.to_csv(path, index=False)
    print(f"[data_loader] Saved {len(df)} records -> {path}")
    return path


if __name__ == "__main__":
    save_clean_csv()
