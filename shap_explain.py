"""SHAP explanations for the saved sklearn pipeline (tree, linear, or kernel)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from generate_dataset import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

FEATURE_LABELS = {
    "pregnancies": "Pregnancies",
    "glucose": "Fasting glucose (mg/dL)",
    "blood_pressure": "Blood pressure (mmHg)",
    "skin_thickness": "Skin thickness (mm)",
    "insulin": "Insulin (µU/mL)",
    "bmi": "BMI",
    "diabetes_pedigree": "Family history score",
    "age": "Age",
    "sleep_hours": "Sleep (hours/night)",
    "resting_hr": "Resting heart rate (bpm)",
    "daily_steps": "Daily steps",
    "active_minutes": "Active minutes/day",
    "hrv_ms": "Heart-rate variability (ms)",
}


class RiskExplainer:
    def __init__(self, model_dir: Path | None = None) -> None:
        directory = Path(model_dir) if model_dir else MODEL_DIR
        model_path = directory / "model.pkl"
        bg_path = directory / "shap_background.csv"
        if not model_path.exists():
            raise FileNotFoundError("Trained model missing. Run: python train.py")
        if not bg_path.exists():
            raise FileNotFoundError("SHAP background sample missing. Run: python train.py")

        self.pipeline = joblib.load(model_path)
        self.preprocess = self.pipeline.named_steps["preprocess"]
        self.estimator = self.pipeline.named_steps["model"]
        self.background = pd.read_csv(bg_path)[FEATURE_COLUMNS]
        self.method = "unavailable"
        self.explainer = None
        self.error: str | None = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        try:
            import shap
        except ImportError:
            self.error = "The shap package is not installed."
            return

        name = type(self.estimator).__name__.lower()
        bg_t = self.preprocess.transform(self.background)
        try:
            if any(k in name for k in ("xgb", "forest", "boost", "tree")):
                self.explainer = shap.TreeExplainer(self.estimator)
                self.method = "tree"
            elif "logistic" in name or "linear" in name:
                self.explainer = shap.LinearExplainer(self.estimator, bg_t)
                self.method = "linear"
            else:
                # KernelExplainer is slower; keep the background small.
                small_bg = bg_t if len(bg_t) <= 40 else bg_t[:40]
                self.explainer = shap.KernelExplainer(
                    self.estimator.predict_proba, small_bg
                )
                self.method = "kernel"
        except Exception as exc:  # SHAP APIs vary by model/version
            self.error = f"Could not create a SHAP explainer for {type(self.estimator).__name__}: {exc}"
            self.explainer = None
            self.method = "unavailable"

    def available(self) -> bool:
        return self.explainer is not None

    def _row(self, features: dict[str, Any] | pd.Series) -> pd.DataFrame:
        if isinstance(features, dict):
            return pd.DataFrame([features])[FEATURE_COLUMNS]
        return features.to_frame().T[FEATURE_COLUMNS]

    def explain(self, features: dict[str, Any] | pd.Series) -> pd.DataFrame:
        if not self.available():
            raise RuntimeError(self.error or "SHAP explainer is not available.")

        row = self._row(features)
        transformed = self.preprocess.transform(row)
        values = self.explainer.shap_values(transformed)

        if isinstance(values, list):
            shap_vals = np.asarray(values[1][0])
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                shap_vals = arr[0, :, 1]
            elif arr.ndim == 2:
                shap_vals = arr[0]
            else:
                shap_vals = arr

        out = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "label": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
                "value": row.iloc[0].to_numpy(),
                "shap": shap_vals,
                "abs_shap": np.abs(shap_vals),
            }
        ).sort_values("abs_shap", ascending=False)
        out["direction"] = np.where(
            out["shap"] > 0,
            "contributed to a higher predicted risk",
            "contributed to a lower predicted risk",
        )
        return out.reset_index(drop=True)

    def global_importance(self, max_rows: int = 80) -> pd.DataFrame:
        if not self.available():
            raise RuntimeError(self.error or "SHAP explainer is not available.")
        sample = self.background.head(max_rows)
        transformed = self.preprocess.transform(sample)
        values = self.explainer.shap_values(transformed)
        if isinstance(values, list):
            arr = np.asarray(values[1])
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:
                arr = arr[:, :, 1]
        mean_abs = np.mean(np.abs(arr), axis=0)
        out = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "label": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return out.reset_index(drop=True)

    def narrative(self, explanation: pd.DataFrame, top_k: int = 5) -> list[str]:
        lines: list[str] = []
        for _, row in explanation.head(top_k).iterrows():
            lines.append(
                f"{row['label']} = {row['value']:g} {row['direction']} "
                f"(SHAP value {row['shap']:+.3f}). This describes the model's "
                "prediction, not a proven medical cause."
            )
        return lines
