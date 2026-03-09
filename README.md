# AI-Driven Collections and Recovery Automation

An intelligent debt-collection prototype that uses **machine learning** to predict repayment probability, segment borrowers by risk, recommend collection actions, simulate communications, enforce compliance rules, and present everything in an interactive dashboard.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Objectives](#objectives)
5. [Architecture](#architecture)
6. [Project Structure](#project-structure)
7. [Technology Stack](#technology-stack)
8. [Methodology](#methodology)
9. [Algorithms Used](#algorithms-used)
10. [Quick Start](#quick-start)
11. [Running Individual Components](#running-individual-components)
12. [API Reference](#api-reference)
13. [Deployment](#deployment)
14. [Results](#results)
15. [Future Work](#future-work)
16. [Conclusion](#conclusion)
17. [Diagram Prompts](#diagram-prompts)

---

## Abstract

Traditional debt-collection processes rely on manual prioritisation and static rules, leading to inefficiency and regulatory risk. This project demonstrates an **AI-driven approach** that generates a synthetic borrower dataset, trains classification models (Logistic Regression, Random Forest, XGBoost), segments debtors by predicted repayment probability, recommends collection strategies, simulates multi-channel communications, and applies compliance guardrails — all orchestrated through a single pipeline and surfaced via a Streamlit dashboard and FastAPI backend.

---

## Introduction

Debt collection is a critical function in financial institutions. Delayed or missed repayments affect cash flow, provisioning, and overall portfolio health. Modern AI/ML techniques can significantly improve recovery rates by:

- **Predicting** which borrowers are likely to repay and which are not.
- **Prioritising** collection efforts toward the highest-value, highest-risk accounts.
- **Personalising** communication to improve engagement.
- **Automating** compliance checks to reduce regulatory exposure.

This project packages all of the above into a runnable prototype.

---

## Problem Statement

Manual debt-collection systems suffer from:
- Inefficient allocation of collection resources.
- One-size-fits-all communication strategies.
- Compliance violations due to lack of automated checks.
- No data-driven prioritisation of borrower accounts.

---

## Objectives

1. Generate a realistic synthetic borrower dataset (5 000+ records).
2. Engineer risk-relevant features and preprocess data.
3. Train and evaluate multiple ML models for repayment prediction.
4. Segment borrowers into Low / Medium / High risk.
5. Map segments to actionable collection strategies.
6. Simulate multi-channel communications.
7. Enforce regulatory compliance rules.
8. Persist all results in a SQLite database.
9. Provide an interactive Streamlit dashboard.
10. Expose a REST API (FastAPI).

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     run_project.py (Orchestrator)                  │
├────────┬────────┬──────────┬──────────┬──────────┬────────────────┤
│ Dataset│Preproc.│ ML Train │ Segment  │ Strategy │ Compliance     │
│ Gen    │Pipeline│ & Eval   │          │ Engine   │ Engine         │
├────────┴────────┴──────────┴──────────┴──────────┴────────────────┤
│                 Communication Simulation Module                    │
├───────────────────────────────────────────────────────────────────┤
│                    SQLite Database Layer                           │
├───────────────────┬───────────────────────────────────────────────┤
│ Streamlit Dashboard│          FastAPI REST Backend                 │
└───────────────────┴───────────────────────────────────────────────┘
```

---

## Project Structure

```
ai_collections_project/
│
├── data/
│   ├── raw/                  # Original synthetic dataset (borrowers.csv)
│   └── processed/            # Preprocessed data, segments, actions, logs
│
├── models/                   # Trained model (.pkl), scaler, metrics, plots
│
├── src/
│   ├── dataset_generator/    # Synthetic data creation
│   │   └── generate.py
│   ├── preprocessing/        # Cleaning, encoding, feature engineering
│   │   └── preprocess.py
│   ├── ml_models/            # Model training & evaluation
│   │   └── train.py
│   ├── segmentation/         # Risk segmentation logic
│   │   └── segmenter.py
│   ├── strategy_engine/      # Rule-based action recommendations
│   │   └── strategy.py
│   ├── compliance_engine/    # Regulatory compliance checks
│   │   └── compliance.py
│   ├── communication_module/ # Message generation & logging
│   │   └── communicate.py
│   ├── database/             # SQLAlchemy models & DB population
│   │   └── models.py
│   └── api/                  # FastAPI REST endpoints
│       └── main.py
│
├── dashboard/
│   └── app.py                # Streamlit multi-page dashboard
│
├── scripts/                  # Standalone convenience scripts
│   ├── generate_dataset.py
│   └── train_model.py
│
├── requirements.txt
├── run_project.py            # One-click full pipeline
└── README.md                 # This file
```

### Folder Descriptions

| Folder | Purpose |
|---|---|
| `data/raw/` | Stores the original generated borrower CSV |
| `data/processed/` | Stores preprocessed data, risk segments, action plans, communication logs, compliance flags |
| `models/` | Persisted ML model, scaler, metrics JSON, feature columns, evaluation plots |
| `src/dataset_generator/` | Code to create the 5 000-row synthetic dataset |
| `src/preprocessing/` | Missing-value imputation, encoding, scaling, feature engineering |
| `src/ml_models/` | Training Logistic Regression, Random Forest, XGBoost; evaluation & selection |
| `src/segmentation/` | Assigns Low / Medium / High risk based on predicted probability |
| `src/strategy_engine/` | Maps risk segments to recommended collection actions |
| `src/compliance_engine/` | Enforces contact-hour, max-attempts, vulnerability, and aggression rules |
| `src/communication_module/` | Generates SMS / Email / Chatbot messages from templates |
| `src/database/` | SQLAlchemy ORM models; populates SQLite from pipeline CSVs |
| `src/api/` | FastAPI REST service exposing borrowers, predictions, actions, compliance |
| `dashboard/` | Streamlit interactive dashboard (5 pages) |
| `scripts/` | Standalone scripts for dataset generation and model training |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib |
| Database | SQLite via SQLAlchemy |
| Dashboard | Streamlit |
| API | FastAPI + Uvicorn |
| Serialisation | Joblib |

---

## Methodology

1. **Data Generation** – Realistic synthetic borrower data with correlated features and a probabilistic target variable.
2. **Preprocessing** – Imputation, one-hot encoding, standard scaling. Three engineered features: `debt_to_income_ratio`, `payment_behavior_score`, `delinquency_index`.
3. **Model Training** – Three classifiers trained on an 80/20 stratified split. Evaluated on Accuracy, Precision, Recall, F1, ROC-AUC. Best model saved.
4. **Segmentation** – Predicted probability thresholds define Low / Medium / High risk.
5. **Strategy** – Rule table maps segment → action (SMS → call → field escalation).
6. **Compliance** – Four rules checked per borrower before contact.
7. **Communication** – Template-based messages generated and logged.
8. **Storage** – All results written to five SQLite tables.
9. **Dashboard** – Five-page Streamlit app reading from the database.
10. **API** – Five FastAPI endpoints for programmatic access.

---

## Algorithms Used

| Algorithm | Role |
|---|---|
| **Logistic Regression** | Baseline linear classifier for recovery prediction |
| **Random Forest** | Ensemble tree model capturing non-linear interactions |
| **XGBoost** | Gradient-boosted trees for maximum predictive accuracy |
| **StandardScaler** | Z-score normalisation for numeric features |
| **One-Hot Encoding** | Categorical → binary dummy variables |

---

## Quick Start

```bash
# 1. Clone / download the project
cd ai_collections_project

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (generates data, trains models, populates DB)
python run_project.py

# 5. Launch the Streamlit dashboard
streamlit run dashboard/app.py

# 6. (Optional) Start the FastAPI server
uvicorn src.api.main:app --reload
```

The dashboard opens at **http://localhost:8501** and the API at **http://localhost:8000/docs**.

---

## Running Individual Components

```bash
# Generate dataset only
python scripts/generate_dataset.py

# Preprocess + train models
python scripts/train_model.py

# Run any module directly
python -m src.segmentation.segmenter
python -m src.strategy_engine.strategy
python -m src.compliance_engine.compliance
python -m src.communication_module.communicate
python -m src.database.models
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/borrowers?skip=0&limit=50` | Paginated borrower list |
| GET | `/borrower/{id}` | Single borrower with risk score and action |
| POST | `/predict` | Predict recovery probability from feature dict |
| GET | `/actions?limit=50` | Collection actions sorted by priority |
| GET | `/compliance?fail_only=true` | Compliance flags (optionally failures only) |

### Predict example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "credit_score": 650, "monthly_income": 50000, "days_past_due": 30}}'
```

---

## Deployment

### Streamlit Cloud

1. Push the project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect the repo and set the main file to `dashboard/app.py`.
4. Ensure `run_project.py` runs during build or pre-generate the database and commit it.

### Render

1. Create a new **Web Service** on [render.com](https://render.com).
2. Set the build command: `pip install -r requirements.txt && python run_project.py`.
3. Set the start command: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`.

### Railway

1. Create a new project on [railway.app](https://railway.app).
2. Add a `Procfile`:
   ```
   web: streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
   ```
3. Add a build command in settings: `python run_project.py`.

For the **FastAPI** backend, use instead:
```
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

---

## Results

After running the pipeline you will see:

- **Model comparison** (example metrics printed to console and saved in `models/metrics.json`):
  - Logistic Regression: ~0.78 AUC
  - Random Forest: ~0.86 AUC
  - XGBoost: ~0.87 AUC (selected as best)
- **Risk distribution**: roughly 35% Low Risk, 30% Medium Risk, 35% High Risk.
- **Compliance flags**: ~15-25% of borrowers flagged depending on time of day.
- **Evaluation plots**: confusion matrices, ROC curves, and feature importance saved in `models/`.

*(Exact numbers depend on the random seed and current hour for compliance checks.)*

---

## Future Work

- **Real data integration** – Replace synthetic data with anonymised production data.
- **Advanced models** – LightGBM, neural networks, survival analysis.
- **Reinforcement learning** – Optimise contact timing and channel selection dynamically.
- **NLP** – Analyse borrower communication sentiment to adapt messaging.
- **Real-time scoring** – Stream new loan events through Apache Kafka.
- **A/B testing** – Compare strategy effectiveness in production.
- **Multi-language** – Support regional language templates for SMS/email.
- **Cloud deployment** – Containerise with Docker and deploy on AWS / Azure / GCP.

---

## Conclusion

This project demonstrates a complete, end-to-end AI-driven collections and recovery system — from data generation through ML modelling, risk segmentation, strategy recommendation, compliance enforcement, communication simulation, and interactive visualisation.  The modular architecture makes it straightforward to swap in real data, plug in additional models, or extend to production-scale infrastructure.

---

## Diagram Prompts

Use these prompts with any diagramming tool (draw.io, Mermaid, Lucidchart, or an AI image generator):

### System Architecture Diagram
> *Draw a system architecture diagram for an AI-driven debt-collection platform. Show: a Data Layer (CSV files, SQLite DB), a Processing Layer (Dataset Generator, Preprocessing Pipeline, ML Training Module), an Intelligence Layer (Risk Segmentation, Strategy Engine, Compliance Engine, Communication Module), and a Presentation Layer (Streamlit Dashboard, FastAPI Backend). Use arrows to show data flow between components.*

### ER Diagram
> *Draw an Entity-Relationship diagram with five tables: borrowers (PK: borrower_id), risk_scores (PK: borrower_id, FK → borrowers), collection_actions (PK: borrower_id, FK → borrowers), communication_logs (PK: log_id, FK → borrowers), compliance_flags (PK: borrower_id, FK → borrowers). Show column names and data types.*

### Use Case Diagram
> *Draw a UML Use Case diagram. Actors: Data Engineer, ML Engineer, Collections Officer, Compliance Officer, Borrower. Use cases: Generate Dataset, Train Model, View Dashboard, Search Borrower, View Risk Score, Assign Strategy, Check Compliance, Send Communication, Access API.*

### Data Flow Diagram (Level 1)
> *Draw a Level-1 DFD. External entities: Borrower Database, Collections Officer. Processes: 1.0 Generate Data, 2.0 Preprocess, 3.0 Train Model, 4.0 Segment Risk, 5.0 Assign Strategy, 6.0 Check Compliance, 7.0 Send Communication. Data stores: D1 Raw Data, D2 Processed Data, D3 Model Store, D4 SQLite DB. Show data flows between all elements.*

### Sequence Diagram
> *Draw a UML Sequence Diagram for the "Borrower Lookup" use case. Actors/objects: Collections Officer → Streamlit Dashboard → SQLite DB → Risk Scores Table → Collection Actions Table. Show the officer entering a borrower ID, the dashboard querying the DB, retrieving risk score and recommended action, and rendering the result.*

---

*Built as an educational prototype. Not intended for production use without further security hardening and real-data validation.*
