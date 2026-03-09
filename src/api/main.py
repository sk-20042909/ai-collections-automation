"""
FastAPI Backend – AI-Driven Collections & Recovery
====================================================
Endpoints
---------
GET  /borrowers          – list all borrowers (paginated)
GET  /borrower/{id}      – single borrower detail + risk + action
POST /predict            – predict recovery probability for a payload
GET  /actions            – list collection actions (sorted by priority)
GET  /compliance         – list compliance flags (optionally filter FAIL only)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# ── Paths ────────────────────────────────────────────────────────────────
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(API_DIR, "..", "..")
DB_PATH = os.path.join(PROJECT_DIR, "data", "collections.db")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.pkl")
COLS_PATH = os.path.join(PROJECT_DIR, "models", "feature_columns.json")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

app = FastAPI(
    title="AI Collections API",
    description="REST API for the AI-Driven Collections & Recovery system.",
    version="1.0.0",
)

# ── Load model once ──────────────────────────────────────────────────────
_model = None
_feature_cols = None


def _get_model():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(COLS_PATH) as f:
            _feature_cols = json.load(f)
    return _model, _feature_cols


# ── Pydantic schemas ─────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Accepts a dictionary of feature values for a single borrower."""
    features: dict


class PredictResponse(BaseModel):
    repayment_probability: float
    risk_segment: str


# ── Helpers ──────────────────────────────────────────────────────────────

def _query(sql: str) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(sql))
        cols = rows.keys()
        return [dict(zip(cols, r)) for r in rows.fetchall()]


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/borrowers")
def list_borrowers(skip: int = 0, limit: int = Query(default=50, le=500)):
    rows = _query(f"SELECT * FROM borrowers LIMIT {int(limit)} OFFSET {int(skip)}")
    return {"count": len(rows), "data": rows}


@app.get("/borrower/{borrower_id}")
def get_borrower(borrower_id: str):
    b = _query(f"SELECT * FROM borrowers WHERE borrower_id = '{borrower_id}'")
    if not b:
        raise HTTPException(404, "Borrower not found")
    r = _query(f"SELECT * FROM risk_scores WHERE borrower_id = '{borrower_id}'")
    a = _query(f"SELECT * FROM collection_actions WHERE borrower_id = '{borrower_id}'")
    return {"borrower": b[0], "risk": r[0] if r else None, "action": a[0] if a else None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, cols = _get_model()
    assert cols is not None, "Feature columns not loaded"
    row = {c: 0 for c in cols}
    row.update(req.features)
    X = pd.DataFrame([row])[cols]
    prob = float(model.predict_proba(X)[0, 1])
    if prob > 0.75:
        seg = "Low Risk"
    elif prob >= 0.45:
        seg = "Medium Risk"
    else:
        seg = "High Risk"
    return PredictResponse(repayment_probability=round(prob, 4), risk_segment=seg)


@app.get("/actions")
def list_actions(limit: int = Query(default=50, le=500)):
    rows = _query(
        f"SELECT ca.*, rs.repayment_probability, rs.risk_segment "
        f"FROM collection_actions ca "
        f"LEFT JOIN risk_scores rs ON ca.borrower_id = rs.borrower_id "
        f"ORDER BY ca.priority_rank ASC LIMIT {int(limit)}"
    )
    return {"count": len(rows), "data": rows}


@app.get("/compliance")
def list_compliance(fail_only: bool = False):
    where = "WHERE compliance_status = 'FAIL'" if fail_only else ""
    rows = _query(f"SELECT * FROM compliance_flags {where}")
    return {"count": len(rows), "data": rows}
