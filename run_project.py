"""
run_project.py – Master Orchestrator
======================================
Runs the entire pipeline end-to-end:
    1. Generate synthetic dataset
    2. Preprocess & feature engineering
    3. Train ML models
    4. Segment borrowers by risk
    5. Recommend collection strategies
    6. Check compliance
    7. Simulate communications
    8. Populate SQLite database

After running this script the Streamlit dashboard and FastAPI server
can be started:
    streamlit run dashboard/app.py
    uvicorn src.api.main:app --reload
"""

import os
import sys
import time

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    start = time.time()

    print("=" * 60)
    print("  AI-Driven Collections & Recovery Automation")
    print("  Full Pipeline Execution")
    print("=" * 60)

    # Step 1 – Generate dataset
    print("\n[1/8] Generating synthetic borrower dataset …")
    from src.dataset_generator.generate import save_dataset
    save_dataset()

    # Step 2 – Preprocess
    print("\n[2/8] Preprocessing & feature engineering …")
    from src.preprocessing.preprocess import run_pipeline
    run_pipeline()

    # Step 3 – Train ML models
    print("\n[3/8] Training ML models …")
    from src.ml_models.train import train_all
    train_all()

    # Step 4 – Risk segmentation
    print("\n[4/8] Segmenting borrowers by risk …")
    from src.segmentation.segmenter import save_segments
    save_segments()

    # Step 5 – Collection strategy
    print("\n[5/8] Recommending collection strategies …")
    from src.strategy_engine.strategy import save_actions
    save_actions()

    # Step 6 – Compliance check
    print("\n[6/8] Running compliance checks …")
    from src.compliance_engine.compliance import save_compliance
    save_compliance()

    # Step 7 – Communication simulation
    print("\n[7/8] Simulating communication messages …")
    from src.communication_module.communicate import save_communication_logs
    save_communication_logs()

    # Step 8 – Populate database
    print("\n[8/8] Populating SQLite database …")
    from src.database.models import populate_all
    populate_all()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print("=" * 60)
    print("\nNext steps:")
    print("  streamlit run dashboard/app.py")
    print("  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
