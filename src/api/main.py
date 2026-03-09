"""
FastAPI – Collections & Recovery API
=====================================
Endpoints:
    GET  /                   – health check
    GET  /borrowers          – paginated borrower list
    GET  /borrower/{id}      – single borrower detail + risk + strategy
    GET  /risk-distribution  – risk-tier counts
    GET  /model-metrics      – all model evaluation metrics
    GET  /compliance-flags   – borrowers with active flags
"""

import os
import json
from fastapi import FastAPI, HTTPException, Query

import pandas as pd

app = FastAPI(title="AI Collections & Recovery API", version="2.0")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found – run the pipeline first.")
    return pd.read_csv(path)


@app.get("/")
def health():
    return {"status": "ok", "version": "2.0", "dataset": "UCI Credit Card Default"}


@app.get("/borrowers")
def list_borrowers(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    df = _read_csv("borrowers_processed.csv")
    page = df.iloc[skip : skip + limit]
    return {"total": len(df), "skip": skip, "limit": limit,
            "data": page.to_dict(orient="records")}


@app.get("/borrower/{borrower_id}")
def get_borrower(borrower_id: str):
    df = _read_csv("borrowers_processed.csv")
    row = df[df["borrower_id"] == borrower_id]
    if row.empty:
        raise HTTPException(404, f"Borrower {borrower_id} not found.")
    info = row.iloc[0].to_dict()

    # Risk
    risk_df = _read_csv("risk_segments.csv")
    risk = risk_df[risk_df["borrower_id"] == borrower_id]
    info["risk"] = risk.iloc[0].to_dict() if not risk.empty else None

    # Strategy
    strat_df = _read_csv("collection_strategies.csv")
    strat = strat_df[strat_df["borrower_id"] == borrower_id]
    info["strategy"] = strat.iloc[0].to_dict() if not strat.empty else None

    return info


@app.get("/risk-distribution")
def risk_distribution():
    df = _read_csv("risk_segments.csv")
    counts = df["risk_tier"].value_counts().to_dict()
    return {"distribution": counts, "total": len(df)}


@app.get("/model-metrics")
def model_metrics():
    path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(404, "metrics.json not found.")
    with open(path) as f:
        return json.load(f)


@app.get("/compliance-flags")
def compliance_flags(min_flags: int = Query(1, ge=0)):
    df = _read_csv("compliance_flags.csv")
    flagged = df[df["flag_count"] >= min_flags]
    return {"total_flagged": len(flagged),
            "data": flagged.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
