"""
Pipeline Orchestrator – run_project.py
======================================
Runs the full AI-Collections pipeline end-to-end:
    1. Load & clean UCI Credit Card Default dataset
    2. Feature engineering & preprocessing
    3. Train models + SHAP explainability
    4. Risk segmentation
    5. Strategy assignment
    6. Compliance checks
    7. Communication generation
    8. Populate SQLite database (6 tables)
"""

import time
import os
import sys

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    start = time.time()
    print("=" * 60)
    print("AI-Driven Collections & Recovery Automation")
    print("Dataset: UCI Credit Card Default (30 000 records)")
    print("=" * 60)

    # Step 1 – Load & clean raw data
    print("\n[1/8] Loading UCI dataset …")
    from src.data_loader.data_loader import save_clean_csv
    save_clean_csv()

    # Step 2 – Preprocess & engineer features
    print("\n[2/8] Preprocessing & feature engineering …")
    from src.preprocessing.preprocess import run_pipeline
    run_pipeline()

    # Step 3 – Train models + SHAP
    print("\n[3/8] Training models (LR, RF, GB, XGB) + SHAP …")
    from src.ml_models.train import train_all
    train_all()

    # Step 4 – Segment borrowers
    print("\n[4/8] Risk segmentation …")
    from src.segmentation.segmenter import segment
    segment()

    # Step 5 – Assign strategies
    print("\n[5/8] Strategy assignment …")
    from src.strategy_engine.strategy import run_strategy
    run_strategy()

    # Step 6 – Compliance checks
    print("\n[6/8] Compliance checks …")
    from src.compliance_engine.compliance import run_compliance
    run_compliance()

    # Step 7 – Communication generation
    print("\n[7/8] Generating outreach messages …")
    from src.communication_module.communicate import run_communication
    run_communication()

    # Step 8 – Populate database
    print("\n[8/8] Populating SQLite database …")
    from src.database.models import populate_all
    populate_all()

    elapsed = round(time.time() - start, 1)
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed}s")
    print("=" * 60)
    print("\nNext steps:")
    print("  Dashboard : python -m streamlit run dashboard/app.py")
    print("  API       : uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
