"""
Database Models (SQLAlchemy + SQLite)
=====================================
Six tables:
    borrowers           – cleaned UCI data
    risk_scores         – default probability & tier
    collection_actions  – strategy assignments
    communication_logs  – outreach messages
    compliance_flags    – compliance check results
    model_metrics       – per-model training metrics
"""

import os
import json
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH = os.path.join(BASE_DIR, "data", "collections.db")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)


# ---------------------------------------------------------------- tables ---

class Borrower(Base):
    __tablename__ = "borrowers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id = Column(String(20), unique=True, nullable=False)
    credit_limit = Column(Float)
    gender = Column(Integer)
    education = Column(Integer)
    marital_status = Column(Integer)
    age = Column(Integer)
    default = Column(Integer)


class RiskScore(Base):
    __tablename__ = "risk_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id = Column(String(20), nullable=False)
    default_probability = Column(Float)
    risk_tier = Column(String(30))
    priority_score = Column(Float)
    scored_at = Column(DateTime, default=datetime.utcnow)


class CollectionAction(Base):
    __tablename__ = "collection_actions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id = Column(String(20), nullable=False)
    risk_tier = Column(String(30))
    recommended_action = Column(Text)
    channel = Column(String(50))
    urgency = Column(String(20))
    follow_up_days = Column(Integer)
    assigned_at = Column(DateTime, default=datetime.utcnow)


class CommunicationLog(Base):
    __tablename__ = "communication_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id = Column(String(20), nullable=False)
    risk_tier = Column(String(30))
    channel = Column(String(50))
    template = Column(String(40))
    message = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow)


class ComplianceFlag(Base):
    __tablename__ = "compliance_flags"
    id = Column(Integer, primary_key=True, autoincrement=True)
    borrower_id = Column(String(20), nullable=False)
    risk_tier = Column(String(30))
    flags = Column(Text)
    flag_count = Column(Integer)
    checked_at = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(60), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)


# ----------------------------------------------------------- population ---

def create_tables():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("[database] Tables dropped & recreated.")


def _clear(table_cls):
    session = Session()
    session.query(table_cls).delete()
    session.commit()
    session.close()


def populate_borrowers():
    df = pd.read_csv(os.path.join(DATA_DIR, "borrowers_processed.csv"))
    _clear(Borrower)
    records = []
    for _, row in df.iterrows():
        records.append(Borrower(
            borrower_id=row["borrower_id"],
            credit_limit=row.get("credit_limit"),
            gender=int(row.get("gender", 0)),
            education=int(row.get("education", 0)),
            marital_status=int(row.get("marital_status", 0)),
            age=int(row.get("age", 0)),
            default=int(row.get("default", 0)),
        ))
    session = Session()
    session.bulk_save_objects(records)
    session.commit()
    session.close()
    print(f"[database] borrowers: {len(records)} rows")


def populate_risk_scores():
    df = pd.read_csv(os.path.join(DATA_DIR, "risk_segments.csv"))
    _clear(RiskScore)
    records = [
        RiskScore(
            borrower_id=row["borrower_id"],
            default_probability=row["default_probability"],
            risk_tier=row["risk_tier"],
            priority_score=row["priority_score"],
        )
        for _, row in df.iterrows()
    ]
    session = Session()
    session.bulk_save_objects(records)
    session.commit()
    session.close()
    print(f"[database] risk_scores: {len(records)} rows")


def populate_collection_actions():
    df = pd.read_csv(os.path.join(DATA_DIR, "collection_strategies.csv"))
    _clear(CollectionAction)
    records = [
        CollectionAction(
            borrower_id=row["borrower_id"],
            risk_tier=row["risk_tier"],
            recommended_action=row["recommended_action"],
            channel=row["channel"],
            urgency=row["urgency"],
            follow_up_days=int(row["follow_up_days"]),
        )
        for _, row in df.iterrows()
    ]
    session = Session()
    session.bulk_save_objects(records)
    session.commit()
    session.close()
    print(f"[database] collection_actions: {len(records)} rows")


def populate_communication_logs():
    df = pd.read_csv(os.path.join(DATA_DIR, "communication_log.csv"))
    _clear(CommunicationLog)
    records = [
        CommunicationLog(
            borrower_id=row["borrower_id"],
            risk_tier=row["risk_tier"],
            channel=row["channel"],
            template=row["template"],
            message=row["message"],
        )
        for _, row in df.iterrows()
    ]
    session = Session()
    session.bulk_save_objects(records)
    session.commit()
    session.close()
    print(f"[database] communication_logs: {len(records)} rows")


def populate_compliance_flags():
    df = pd.read_csv(os.path.join(DATA_DIR, "compliance_flags.csv"))
    _clear(ComplianceFlag)
    records = [
        ComplianceFlag(
            borrower_id=row["borrower_id"],
            risk_tier=row["risk_tier"],
            flags=row["flags"],
            flag_count=int(row["flag_count"]),
        )
        for _, row in df.iterrows()
    ]
    session = Session()
    session.bulk_save_objects(records)
    session.commit()
    session.close()
    print(f"[database] compliance_flags: {len(records)} rows")


def populate_model_metrics():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        print("[database] metrics.json not found – skipping model_metrics.")
        return
    with open(metrics_path) as f:
        metrics_list = json.load(f)
    _clear(ModelMetric)
    records = [
        ModelMetric(
            model_name=m["model"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1_score=m["f1_score"],
            roc_auc=m["roc_auc"],
        )
        for m in metrics_list
    ]
    session = Session()
    session.bulk_save_objects(records)
    session.commit()
    session.close()
    print(f"[database] model_metrics: {len(records)} rows")


def populate_all():
    create_tables()
    populate_borrowers()
    populate_risk_scores()
    populate_collection_actions()
    populate_communication_logs()
    populate_compliance_flags()
    populate_model_metrics()
    print("[database] All 6 tables populated ✓")


if __name__ == "__main__":
    populate_all()
