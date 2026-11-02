"""Model comparison helpers and evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


SCREENING_NOTE = (
    "For a screening tool, missing a potentially high-risk case (false negative) "
    "is typically more costly than a false alarm. That is why recall (sensitivity) "
    "and ROC-AUC are considered alongside accuracy. Accuracy alone can look strong "
    "when the negative class is larger."
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, Any]:
    """Compute holdout metrics from predicted probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fpr, tpr, roc_thresholds = roc_curve(y_true, proba)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "decision_threshold": threshold,
    }


def stratified_cv_scores(pipeline, x, y, seed: int = 42, n_splits: int = 5) -> dict[str, float]:
    """Stratified k-fold scores used for model selection (train fold only)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    results = cross_validate(
        pipeline,
        x,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    return {f"cv_{name}": float(np.mean(results[f"test_{name}"])) for name in scoring}


def select_best_model(cv_rows: list[dict[str, Any]]) -> str:
    """Pick a screening-oriented winner: highest CV ROC-AUC, then recall, then F1."""
    ranked = sorted(
        cv_rows,
        key=lambda row: (row["cv_roc_auc"], row["cv_recall"], row["cv_f1"]),
        reverse=True,
    )
    return str(ranked[0]["model"])


def clone_and_fit(pipeline, x, y):
    fitted = clone(pipeline)
    fitted.fit(x, y)
    return fitted
