"""
Database Layer – SQLAlchemy ORM + SQLite
=========================================
Defines five tables and provides helpers to populate them from CSV outputs.

Tables
------
- borrowers           : Raw borrower attributes
- risk_scores         : Model predictions & risk segments
- collection_actions  : Recommended actions per borrower
- communication_logs  : Simulated contact messages
- compliance_flags    : Compliance check results
"""

import os
import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH = os.path.join(BASE_DIR, "data", "collections.db")
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ───────────────────────────────── Models ─────────────────────────────────

class Borrower(Base):
    __tablename__ = "borrowers"
    borrower_id = Column(String(20), primary_key=True)
    age = Column(Integer)
    employment_status = Column(String(30))
    monthly_income = Column(Float)
    credit_score = Column(Integer)
    loan_amount = Column(Float)
    interest_rate = Column(Float)
    loan_tenure_months = Column(Integer)
    emi_amount = Column(Float)
    days_past_due = Column(Integer)
    number_of_missed_payments = Column(Integer)
    last_payment_amount = Column(Float)
    contact_response_rate = Column(Float)
    prior_collection_attempts = Column(Integer)
    preferred_contact_channel = Column(String(20))
    outstanding_balance = Column(Float)
    region = Column(String(20))
    customer_tenure = Column(Integer)
    repayment_status = Column(String(20))
    recovered = Column(Integer)


class RiskScore(Base):
    __tablename__ = "risk_scores"
    borrower_id = Column(String(20), primary_key=True)
    repayment_probability = Column(Float)
    risk_segment = Column(String(20))
    priority_score = Column(Float)


class CollectionAction(Base):
    __tablename__ = "collection_actions"
    borrower_id = Column(String(20), primary_key=True)
    recommended_action = Column(String(50))
    urgency_level = Column(String(20))
    priority_rank = Column(Integer)


class CommunicationLog(Base):
    __tablename__ = "communication_logs"
    log_id = Column(String(40), primary_key=True)
    borrower_id = Column(String(20))
    channel = Column(String(20))
    message = Column(Text)
    timestamp = Column(String(30))
    status = Column(String(20))


class ComplianceFlag(Base):
    __tablename__ = "compliance_flags"
    borrower_id = Column(String(20), primary_key=True)
    risk_segment = Column(String(20))
    recommended_action = Column(String(50))
    compliance_status = Column(String(10))
    violation_reason = Column(Text)


# ─────────────────────────── Helper Functions ─────────────────────────────

def init_db():
    """Create all tables (idempotent)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(engine)
    print(f"[database] Initialized SQLite DB at {DB_PATH}")


def _upsert_df(df: pd.DataFrame, table_name: str):
    """Write a DataFrame into the given table (replace strategy)."""
    df.to_sql(table_name, engine, if_exists="replace", index=False)


def populate_all():
    """Load all CSV outputs into the database."""
    init_db()

    data_dir = os.path.join(BASE_DIR, "data")

    mappings = {
        "borrowers": os.path.join(data_dir, "raw", "borrowers.csv"),
        "risk_scores": os.path.join(data_dir, "processed", "risk_segments.csv"),
        "collection_actions": os.path.join(data_dir, "processed", "collection_actions.csv"),
        "communication_logs": os.path.join(data_dir, "processed", "communication_logs.csv"),
        "compliance_flags": os.path.join(data_dir, "processed", "compliance_flags.csv"),
    }

    for table, csv_path in mappings.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Only keep columns that match the expected table schema:
            if table == "collection_actions":
                keep = ["borrower_id", "recommended_action", "urgency_level", "priority_rank"]
                df = df[[c for c in keep if c in df.columns]]
            _upsert_df(df, table)
            print(f"[database] Loaded {len(df)} rows -> {table}")
        else:
            print(f"[database] SKIP {table} ({csv_path} not found)")


if __name__ == "__main__":
    populate_all()
