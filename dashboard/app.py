"""
Streamlit Dashboard – AI-Driven Collections & Recovery
=======================================================
Multi-page dashboard:
  1. Overview   – key metrics & recovery distribution
  2. Risk Analysis – segment pie chart & probability histogram
  3. Borrower Lookup – search & display individual borrower detail
  4. Collection Strategy – top priority accounts table
  5. Compliance Alerts – flagged borrowers list
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# ── Resolve paths relative to this file ──────────────────────────────────
DASH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(DASH_DIR, "..")
DB_PATH = os.path.join(PROJECT_DIR, "data", "collections.db")

if not os.path.exists(DB_PATH):
    st.error(
        f"Database not found at `{DB_PATH}`.  "
        "Run `python run_project.py` first to generate all data."
    )
    st.stop()

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

# ── Helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Collections Dashboard", layout="wide")
st.title("🏦 AI-Driven Collections & Recovery Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Risk Analysis", "Borrower Lookup", "Collection Strategy", "Compliance Alerts"],
)

# ==================== 1. OVERVIEW =========================================
if page == "Overview":
    st.header("Overview")

    borrowers = query("SELECT * FROM borrowers")
    risk = query("SELECT * FROM risk_scores")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Borrowers", f"{len(borrowers):,}")
    overdue = borrowers[borrowers["days_past_due"] > 0]
    col2.metric("Overdue Borrowers", f"{len(overdue):,}")
    col3.metric("Avg Credit Score", f"{borrowers['credit_score'].mean():.0f}")
    col4.metric("Recovery Rate", f"{borrowers['recovered'].mean()*100:.1f}%")

    st.subheader("Recovery Probability Distribution")
    fig = px.histogram(
        risk,
        x="repayment_probability",
        nbins=40,
        color_discrete_sequence=["#636EFA"],
        labels={"repayment_probability": "Repayment Probability"},
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Days Past Due Distribution")
    fig2 = px.histogram(
        borrowers,
        x="days_past_due",
        nbins=50,
        color_discrete_sequence=["#EF553B"],
    )
    st.plotly_chart(fig2, width="stretch")

# ==================== 2. RISK ANALYSIS ====================================
elif page == "Risk Analysis":
    st.header("Risk Analysis")

    risk = query("SELECT * FROM risk_scores")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Segment Distribution")
        counts = risk["risk_segment"].value_counts().reset_index()
        counts.columns = ["risk_segment", "count"]
        fig = px.pie(
            counts,
            names="risk_segment",
            values="count",
            color="risk_segment",
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f39c12",
                "High Risk": "#e74c3c",
            },
            hole=0.35,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Probability Histogram by Segment")
        fig2 = px.histogram(
            risk,
            x="repayment_probability",
            color="risk_segment",
            nbins=40,
            color_discrete_map={
                "Low Risk": "#2ecc71",
                "Medium Risk": "#f39c12",
                "High Risk": "#e74c3c",
            },
            barmode="overlay",
            opacity=0.65,
        )
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Segment Summary Statistics")
    summary = (
        risk.groupby("risk_segment")["repayment_probability"]
        .describe()
        .round(3)
    )
    st.dataframe(summary)

# ==================== 3. BORROWER LOOKUP ==================================
elif page == "Borrower Lookup":
    st.header("Borrower Lookup")

    search = st.text_input("Enter Borrower ID (e.g. BRW-00001)")
    if search:
        safe_id = search.strip()
        borrower = query(
            f"SELECT * FROM borrowers WHERE borrower_id = '{safe_id}'"
        )
        risk_row = query(
            f"SELECT * FROM risk_scores WHERE borrower_id = '{safe_id}'"
        )
        action_row = query(
            f"SELECT * FROM collection_actions WHERE borrower_id = '{safe_id}'"
        )

        if borrower.empty:
            st.warning("Borrower not found.")
        else:
            st.subheader("Borrower Profile")
            st.dataframe(borrower.T.rename(columns={borrower.index[0]: "Value"}))

            if not risk_row.empty:
                st.subheader("Risk Assessment")
                c1, c2, c3 = st.columns(3)
                c1.metric("Repayment Probability", f"{risk_row.iloc[0]['repayment_probability']:.2%}")
                c2.metric("Risk Segment", risk_row.iloc[0]["risk_segment"])
                c3.metric("Priority Score", f"{risk_row.iloc[0]['priority_score']:.4f}")

            if not action_row.empty:
                st.subheader("Recommended Action")
                st.info(
                    f"**{action_row.iloc[0]['recommended_action']}** "
                    f"(Urgency: {action_row.iloc[0]['urgency_level']}, "
                    f"Rank: {action_row.iloc[0]['priority_rank']})"
                )

# ==================== 4. COLLECTION STRATEGY ==============================
elif page == "Collection Strategy":
    st.header("Collection Strategy – Top Priority Accounts")

    n = st.slider("Number of accounts to show", 10, 200, 50)
    actions = query(
        f"SELECT * FROM collection_actions ORDER BY priority_rank ASC LIMIT {int(n)}"
    )
    risk = query("SELECT * FROM risk_scores")
    merged = actions.merge(risk, on="borrower_id", how="left")
    st.dataframe(merged, width="stretch", height=500)

# ==================== 5. COMPLIANCE ALERTS ================================
elif page == "Compliance Alerts":
    st.header("Compliance Alerts")

    flags = query("SELECT * FROM compliance_flags WHERE compliance_status = 'FAIL'")
    st.metric("Total Violations", f"{len(flags):,}")

    if not flags.empty:
        st.dataframe(flags, width="stretch", height=500)
    else:
        st.success("No compliance violations found.")
