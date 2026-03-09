"""
Streamlit Dashboard – AI Collections & Recovery  (7 pages)
==========================================================
Pages:
    1. Overview              – KPIs, tier pie chart, probability histogram
    2. Risk Analysis         – heatmap, distribution box-plots
    3. Borrower Explorer     – searchable table with details
    4. Collection Strategy   – strategy breakdown
    5. Compliance Alerts     – flagged borrowers
    6. Model Performance     – accuracy / AUC comparison, confusion matrices
    7. SHAP Explainability   – global & local SHAP plots
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ---------------------------------------------------------------- paths ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = os.path.join(DATA_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []


def load_image(name):
    p = os.path.join(MODELS_DIR, name)
    return Image.open(p) if os.path.exists(p) else None


# ----------------------------------------------------------- page setup ---
st.set_page_config(page_title="AI Collections Dashboard", layout="wide")
st.title("🏦 AI-Driven Collections & Recovery Dashboard")

PAGES = [
    "Overview",
    "Risk Analysis",
    "Borrower Explorer",
    "Collection Strategy",
    "Compliance Alerts",
    "Model Performance",
    "SHAP Explainability",
]
page = st.sidebar.radio("Navigate", PAGES)

# ------------------------------------------------------------- helpers ---

TIER_COLORS = {
    "Low Risk": "#2ecc71",
    "Medium Risk": "#f1c40f",
    "High Risk": "#e67e22",
    "Very High Risk": "#e74c3c",
}


# ============================================================ PAGE 1 ====
def page_overview():
    st.header("📊 Overview")
    segments = load_csv("risk_segments.csv")
    borrowers = load_csv("borrowers_processed.csv")

    if segments.empty:
        st.warning("No data – run the pipeline first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Borrowers", f"{len(borrowers):,}")
    c2.metric("Default Rate",
              f"{borrowers['default'].mean()*100:.1f}%" if "default" in borrowers else "N/A")
    c3.metric("Avg Default Prob",
              f"{segments['default_probability'].mean():.2%}")
    c4.metric("High/V.High Risk",
              f"{len(segments[segments['risk_tier'].isin(['High Risk','Very High Risk'])]):,}")

    col1, col2 = st.columns(2)
    with col1:
        tier_counts = segments["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        fig = px.pie(tier_counts, names="risk_tier", values="count",
                     title="Risk Tier Distribution",
                     color="risk_tier", color_discrete_map=TIER_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(segments, x="default_probability", nbins=50,
                           title="Default Probability Distribution",
                           color_discrete_sequence=["#3498db"])
        fig.add_vline(x=0.30, line_dash="dash", line_color="orange",
                      annotation_text="Medium threshold")
        fig.add_vline(x=0.60, line_dash="dash", line_color="red",
                      annotation_text="High threshold")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================ PAGE 2 ====
def page_risk_analysis():
    st.header("📈 Risk Analysis")
    segments = load_csv("risk_segments.csv")
    borrowers = load_csv("borrowers_processed.csv")
    if segments.empty:
        st.warning("No data.")
        return

    # Safely select only columns that exist in the processed data
    merge_cols = ["borrower_id"] + [c for c in ["age", "credit_limit", "education"]
                                     if c in borrowers.columns]
    merged = segments.merge(borrowers[merge_cols], on="borrower_id", how="left")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(merged, x="risk_tier", y="default_probability",
                     color="risk_tier", color_discrete_map=TIER_COLORS,
                     title="Default Probability by Risk Tier")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(merged, x="age", y="default_probability",
                         color="risk_tier", color_discrete_map=TIER_COLORS,
                         title="Age vs. Default Probability", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(merged, x="risk_tier", y="credit_limit",
                 color="risk_tier", color_discrete_map=TIER_COLORS,
                 title="Credit Limit by Risk Tier")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================ PAGE 3 ====
def page_borrower_explorer():
    st.header("🔍 Borrower Explorer")
    borrowers = load_csv("borrowers_processed.csv")
    segments = load_csv("risk_segments.csv")
    if borrowers.empty:
        st.warning("No data.")
        return

    merged = borrowers.merge(segments, on="borrower_id", how="left")

    search = st.text_input("Search by Borrower ID (e.g. BRW-00001)")
    tier_filter = st.multiselect("Filter by Risk Tier",
                                 ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"])

    if tier_filter:
        merged = merged[merged["risk_tier"].isin(tier_filter)]
    if search:
        merged = merged[merged["borrower_id"].str.contains(search, case=False, na=False)]

    st.dataframe(merged.head(200), use_container_width=True)

    if search and not merged.empty:
        row = merged.iloc[0]
        st.subheader(f"Details for {row['borrower_id']}")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Default Probability", f"{row.get('default_probability', 0):.2%}")
        dc2.metric("Risk Tier", row.get("risk_tier", "N/A"))
        dc3.metric("Priority Score", row.get("priority_score", "N/A"))


# ============================================================ PAGE 4 ====
def page_collection_strategy():
    st.header("📋 Collection Strategy")
    strats = load_csv("collection_strategies.csv")
    if strats.empty:
        st.warning("No data.")
        return

    tier_summary = strats.groupby("risk_tier").agg(
        count=("borrower_id", "count"),
        avg_priority=("priority_score", "mean"),
    ).reset_index()

    fig = px.bar(tier_summary, x="risk_tier", y="count",
                 color="risk_tier", color_discrete_map=TIER_COLORS,
                 title="Accounts per Strategy Tier")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategy Details")
    for tier in ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]:
        subset = strats[strats["risk_tier"] == tier]
        if subset.empty:
            continue
        with st.expander(f"{tier} ({len(subset)} accounts)"):
            row = subset.iloc[0]
            st.markdown(f"**Action:** {row['recommended_action']}")
            st.markdown(f"**Channel:** {row['channel']}")
            st.markdown(f"**Urgency:** {row['urgency']}")
            st.markdown(f"**Follow-up:** every {row['follow_up_days']} days")
            st.markdown(f"**Description:** {row['description']}")


# ============================================================ PAGE 5 ====
def page_compliance_alerts():
    st.header("⚠️ Compliance Alerts")
    flags = load_csv("compliance_flags.csv")
    if flags.empty:
        st.warning("No data.")
        return

    c1, c2 = st.columns(2)
    flagged = flags[flags["flag_count"] > 0]
    c1.metric("Total Flagged", len(flagged))
    c2.metric("Clear", len(flags) - len(flagged))

    fig = px.histogram(flags, x="flag_count", nbins=10,
                       title="Flag Count Distribution",
                       color_discrete_sequence=["#e74c3c"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flagged Borrowers")
    st.dataframe(flagged.head(200), use_container_width=True)


# ============================================================ PAGE 6 ====
def page_model_performance():
    st.header("🤖 Model Performance")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics – run training first.")
        return

    mdf = pd.DataFrame(metrics)
    st.dataframe(mdf, use_container_width=True)

    # Bar chart of AUC
    fig = px.bar(mdf, x="model", y="roc_auc", color="model",
                 title="ROC-AUC by Model",
                 text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    cats = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig = go.Figure()
    for _, row in mdf.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in cats],
            theta=cats,
            fill="toself",
            name=row["model"],
        ))
    fig.update_layout(title="Model Comparison (Radar)", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(len(metrics))
    for idx, m in enumerate(metrics):
        img = load_image(f"{m['model']}_confusion.png")
        if img:
            cols[idx].image(img, caption=m["model"], use_container_width=True)

    # ROC curves
    roc_img = load_image("roc_curves.png")
    if roc_img:
        st.image(roc_img, caption="ROC Curves", use_container_width=True)


# ============================================================ PAGE 7 ====
def page_shap_explainability():
    st.header("🔬 SHAP Explainability")

    shap_summary = load_image("shap_summary.png")
    shap_dot = load_image("shap_dot.png")
    fi_img = load_image("feature_importance.png")

    if shap_summary is None and shap_dot is None:
        st.warning("No SHAP plots found – ensure SHAP ran during training.")
        return

    st.subheader("Global Feature Importance (SHAP Bar)")
    if shap_summary:
        st.image(shap_summary, use_container_width=True)

    st.subheader("SHAP Summary (Dot Plot)")
    if shap_dot:
        st.image(shap_dot, use_container_width=True)

    if fi_img:
        st.subheader("Sklearn Feature Importance (Best Model)")
        st.image(fi_img, use_container_width=True)

    # Show SHAP values table
    shap_csv = os.path.join(MODELS_DIR, "shap_values.csv")
    if os.path.exists(shap_csv):
        st.subheader("SHAP Values (sample)")
        sdf = pd.read_csv(shap_csv)
        st.dataframe(sdf.head(50), use_container_width=True)

        # Top features by mean |SHAP|
        mean_abs = sdf.abs().mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=mean_abs.values, y=mean_abs.index, orientation="h",
                     title="Top 15 Features by Mean |SHAP Value|",
                     labels={"x": "Mean |SHAP|", "y": "Feature"})
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------ routing ---
PAGE_MAP = {
    "Overview": page_overview,
    "Risk Analysis": page_risk_analysis,
    "Borrower Explorer": page_borrower_explorer,
    "Collection Strategy": page_collection_strategy,
    "Compliance Alerts": page_compliance_alerts,
    "Model Performance": page_model_performance,
    "SHAP Explainability": page_shap_explainability,
}

PAGE_MAP[page]()
