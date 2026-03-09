# AI-Driven Collections & Recovery Automation

> End-to-end machine-learning pipeline for credit-card default prediction,
> risk segmentation, collection-strategy assignment, compliance enforcement,
> and borrower outreach — powered by the **UCI Credit Card Default** dataset
> (30 000 real-world records).

---

## Key Features

| Module | Description |
|---|---|
| **Data Loader** | Loads & cleans the raw UCI `.xls` file, standardises column names |
| **Preprocessing** | Missing-value handling, 5 engineered features, one-hot encoding, scaling |
| **ML Training** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **SHAP Explainability** | Global summary plots, dot plots, per-feature SHAP values |
| **Risk Segmentation** | 4 tiers — Low / Medium / High / Very High — with priority scoring |
| **Strategy Engine** | Rule-based + ML-priority channel & action recommendations |
| **Compliance Engine** | Contact-time window, frequency cap, vulnerable-customer flags |
| **Communication** | Personalised SMS, email, chatbot, and payment-assistance templates |
| **Database** | SQLite with 6 tables (borrowers, risk_scores, actions, comms, compliance, model_metrics) |
| **Dashboard** | 7-page Streamlit app with Plotly charts and SHAP visualisations |
| **REST API** | FastAPI with borrower lookup, risk distribution, model metrics |

---

## Dataset

**UCI Credit Card Default** — 30 000 records × 25 features  
Source: <https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients>

Target variable: `default payment next month` → renamed to `default` (binary 0/1).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data too)
python run_project.py

# 3. Launch the dashboard
python -m streamlit run dashboard/app.py

# 4. (Optional) Start the REST API
uvicorn src.api.main:app --reload
```

---

## Project Structure

```
ai_collections_project/
├── data/
│   ├── raw/                    # UCI xls file
│   ├── processed/              # Clean CSVs
│   └── collections.db          # SQLite database
├── models/                     # Trained model, metrics, SHAP plots
├── src/
│   ├── data_loader/            # UCI dataset loader
│   ├── preprocessing/          # Feature engineering & scaling
│   ├── ml_models/              # Training + SHAP + evaluation
│   ├── segmentation/           # Risk-tier assignment
│   ├── strategy_engine/        # Collection-action mapping
│   ├── compliance_engine/      # Regulatory-flag checks
│   ├── communication_module/   # Outreach message generation
│   ├── database/               # SQLAlchemy models & population
│   └── api/                    # FastAPI endpoints
├── dashboard/
│   └── app.py                  # 7-page Streamlit dashboard
├── run_project.py              # End-to-end orchestrator
├── requirements.txt
└── README.md
```

---

## Risk Tiers

| Tier | Default Probability | Strategy |
|---|---|---|
| Low Risk | < 0.30 | Automated SMS / email reminders |
| Medium Risk | 0.30 – 0.60 | Call-center outreach + payment plan |
| High Risk | 0.60 – 0.80 | Escalated recovery + restructured payment |
| Very High Risk | ≥ 0.80 | Legal review / settlement offer |

---

## Models Evaluated

- **Logistic Regression** – interpretable baseline
- **Random Forest** – ensemble with feature importance
- **Gradient Boosting** – sequential boosting
- **XGBoost** – regularised gradient boosting

Best model (by ROC-AUC) is automatically persisted and used downstream.

---

## License

Educational / demonstration purposes only.
