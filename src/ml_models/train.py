"""
ML Model Training & Evaluation + SHAP Explainability
=====================================================
Trains four classifiers to predict credit-card default:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. Gradient Boosting

Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC.
Plots: Confusion matrices, ROC curves, Feature importance, SHAP summary.
The best model (by ROC-AUC) is persisted via Joblib.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    XGBClassifier = None
    HAS_XGB = False

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

HAS_SHAP = shap is not None

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


# ---------------------------------------------------------------- helpers ---

def _load_data(path: str | None = None):
    if path is None:
        path = os.path.join(DATA_DIR, "borrowers_processed.csv")
    df = pd.read_csv(path)
    # Drop ID, target, and preserved original categorical columns (dashboard-only)
    non_feature_cols = ["borrower_id", "gender", "education", "marital_status"]
    drop_cols = [c for c in non_feature_cols if c in df.columns]
    X = df.drop(columns=["default"] + drop_cols, errors="ignore")
    y = df["default"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _evaluate(name, model, X_test, y_test) -> tuple[dict, object, object]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }
    return metrics, y_pred, y_prob


def _plot_confusion(name, y_test, y_pred, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues", alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} – Confusion Matrix")
    path = os.path.join(out_dir, f"{name}_confusion.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(results, X_test, y_test, out_dir):
    fig, ax = plt.subplots()
    for name, model, _, _ in results:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "roc_curves.png"), bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(model, feature_names, out_dir):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return
    idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], imp[idx])
    ax.set_title("Top 15 Feature Importances (Best Model)")
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), bbox_inches="tight")
    plt.close(fig)


def _generate_shap(model, X_train, X_test, out_dir):
    """Generate SHAP summary plot and save SHAP values for dashboard use."""
    if not HAS_SHAP or shap is None:
        print("[ml_models] SHAP not installed – skipping explainability plots.")
        return

    print("[ml_models] Generating SHAP explanations …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use a sample for speed
        sample_size = min(500, len(X_test))
        X_sample = X_test.iloc[:sample_size]

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            bg = shap.sample(X_train, 100)
            explainer = shap.LinearExplainer(model, bg)

        shap_values = explainer.shap_values(X_sample)

        # For tree models shap_values may be a list [class0, class1]
        # or a 3-D array (samples, features, classes)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_summary.png"), bbox_inches="tight")
        plt.close("all")

        # Summary dot plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_dot.png"), bbox_inches="tight")
        plt.close("all")

        # Save SHAP values for dashboard use
        shap_df = pd.DataFrame(sv, columns=X_sample.columns)
        shap_df.to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)
        print(f"[ml_models] SHAP plots & values saved -> {out_dir}")


# ------------------------------------------------------------------ main ---

def train_all(data_path: str | None = None) -> str:
    """Train models, evaluate, plot, generate SHAP, and save the best."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = _load_data(data_path)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }
    if HAS_XGB and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )

    results = []
    for name, clf in models.items():
        print(f"[ml_models] Training {name} …")
        clf.fit(X_train, y_train)
        metrics, y_pred, y_prob = _evaluate(name, clf, X_test, y_test)
        print(f"  -> {metrics}")
        _plot_confusion(name, y_test, y_pred, MODELS_DIR)
        results.append((name, clf, metrics, y_prob))

    _plot_roc(results, X_test, y_test, MODELS_DIR)

    # Best model
    best_name, best_model, best_metrics, _ = max(results, key=lambda r: r[2]["roc_auc"])
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[ml_models] Best: {best_name} (AUC={best_metrics['roc_auc']})")
    print(f"[ml_models] Saved -> {model_path}")

    _plot_feature_importance(best_model, list(X_train.columns), MODELS_DIR)
    _generate_shap(best_model, X_train, X_test, MODELS_DIR)

    # Persist metrics & column order
    all_metrics = [r[2] for r in results]
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X_train.columns), f)

    return model_path


if __name__ == "__main__":
    train_all()
