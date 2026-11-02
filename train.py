"""Train multiple models, select a screening-oriented winner, and save artifacts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from generate_dataset import FEATURE_COLUMNS, TARGET, save_dataset
from model_utils import (
    SCREENING_NOTE,
    classification_metrics,
    clone_and_fit,
    select_best_model,
    stratified_cv_scores,
)
from preprocessing import build_model_pipeline, build_preprocessor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"


def _try_xgboost(random_state: int, scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )


def candidate_estimators(y_train, seed: int = 42) -> dict:
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    xgb = _try_xgboost(seed, scale_pos_weight)
    if xgb is not None:
        models["XGBoost"] = xgb
    return models


def profile_dataset(df: pd.DataFrame) -> dict:
    """Inspect the table so the app can report real data facts, not guesses."""
    feature_df = df[FEATURE_COLUMNS]
    missing = feature_df.isna().sum().to_dict()
    missing[TARGET] = int(df[TARGET].isna().sum())

    numeric = feature_df.select_dtypes(include=[np.number])
    outlier_counts = {}
    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_counts[col] = 0
            continue
        mask = (numeric[col] < q1 - 1.5 * iqr) | (numeric[col] > q3 + 1.5 * iqr)
        outlier_counts[col] = int(mask.sum())

    pos = int((df[TARGET] == 1).sum())
    neg = int((df[TARGET] == 0).sum())
    return {
        "n_rows": int(len(df)),
        "n_features": int(len(FEATURE_COLUMNS)),
        "target": TARGET,
        "numeric_features": FEATURE_COLUMNS,
        "categorical_features": [],
        "missing_values": {k: int(v) for k, v in missing.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {"no_diabetes": neg, "diabetes": pos},
        "positive_prevalence": float(df[TARGET].mean()),
        "iqr_outlier_counts": outlier_counts,
        "possible_data_leakage": (
            "No target column is included in the feature list. Labels were "
            "sampled from a synthetic logistic process of the features plus noise. "
            "That is a constructed relationship for teaching, not leakage from a "
            "real EHR extract. Results will look stronger than on messy clinical data."
        ),
        "data_source": "synthetic_fused_clinical_lifestyle",
        "synthetic": True,
    }


def _save_plots(metrics: dict, comparison: list[dict]) -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black", fontsize=13)
    ax.set_title("Holdout confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    roc = metrics["roc_curve"]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(roc["fpr"], roc["tpr"], color="#0B6E4F", lw=2, label=f"AUC={metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("Holdout ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "roc_curve.png", dpi=140)
    plt.close(fig)

    names = [row["model"] for row in comparison]
    aucs = [row["test_roc_auc"] for row in comparison]
    recalls = [row["test_recall"] for row in comparison]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, aucs, w, label="ROC-AUC", color="#0B6E4F")
    ax.bar(x + w / 2, recalls, w, label="Recall", color="#7EB8A4")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Holdout comparison (not used for selection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "model_comparison.png", dpi=140)
    plt.close(fig)


def train(seed: int = 42) -> dict:
    if not DATA_PATH.exists():
        save_dataset(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    profile = profile_dataset(df)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    estimators = candidate_estimators(y_train, seed=seed)

    cv_rows: list[dict] = []
    test_rows: list[dict] = []
    fitted_models: dict[str, object] = {}

    for name, estimator in estimators.items():
        pipe = build_model_pipeline(estimator, build_preprocessor())
        print(f"Cross-validating {name}...")
        cv_scores = stratified_cv_scores(pipe, x_train, y_train, seed=seed)
        cv_rows.append({"model": name, **cv_scores})

        fitted = clone_and_fit(pipe, x_train, y_train)
        fitted_models[name] = fitted
        proba = fitted.predict_proba(x_test)[:, 1]
        holdout = classification_metrics(y_test, proba)
        test_rows.append(
            {
                "model": name,
                "test_accuracy": holdout["accuracy"],
                "test_precision": holdout["precision"],
                "test_recall": holdout["recall"],
                "test_f1": holdout["f1"],
                "test_roc_auc": holdout["roc_auc"],
            }
        )

    selected_name = select_best_model(cv_rows)
    selected = fitted_models[selected_name]
    selected_proba = selected.predict_proba(x_test)[:, 1]
    selected_metrics = classification_metrics(y_test, selected_proba)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_DIR / "model.pkl")
    joblib.dump(selected.named_steps["preprocess"], MODEL_DIR / "preprocessor.pkl")

    background = x_train.sample(n=min(180, len(x_train)), random_state=seed)
    background.to_csv(MODEL_DIR / "shap_background.csv", index=False)

    comparison = []
    for cv, te in zip(cv_rows, test_rows):
        comparison.append({**cv, **{k: v for k, v in te.items() if k != "model"}})

    _save_plots(selected_metrics, comparison)

    payload = {
        **selected_metrics,
        "selected_model": selected_name,
        "selection_rule": (
            "Highest mean stratified 5-fold ROC-AUC on the training split, "
            "with recall then F1 as tie-breakers. Holdout metrics are reported "
            "separately and were not used to pick the model."
        ),
        "screening_note": SCREENING_NOTE,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "features": FEATURE_COLUMNS,
        "training_date": date.today().isoformat(),
        "comparison": comparison,
        "data_profile": profile,
        "xgboost_available": "XGBoost" in estimators,
    }

    (MODEL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("selected_model", "accuracy", "precision", "recall", "f1", "roc_auc")}, indent=2))
    print("Saved", MODEL_DIR / "model.pkl")
    return payload


if __name__ == "__main__":
    train()
