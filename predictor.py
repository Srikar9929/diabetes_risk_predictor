"""Load the saved pipeline and map probabilities to educational risk bands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from generate_dataset import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# These cut-points only group the model's probability for display.
# They are not clinical diagnostic thresholds.
RISK_BANDS = (
    (0.33, "Low Risk"),
    (0.66, "Moderate Risk"),
    (1.01, "High Risk"),
)

BAND_EXPLANATION = (
    "Risk categories are simple bins of the model's predicted probability: "
    "Low < 33%, Moderate 33% to 66%, High ≥ 66%. They are for communication "
    "in this educational app and are not medically validated cut-offs."
)


class RiskPredictor:
    def __init__(self, model_dir: Path | None = None) -> None:
        directory = Path(model_dir) if model_dir else MODEL_DIR
        model_path = directory / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing trained model at {model_path}. From the project folder run: python train.py"
            )
        self.model = joblib.load(model_path)
        self.feature_columns = FEATURE_COLUMNS
        self.model_path = model_path

    def as_frame(self, features: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, pd.Series):
            frame = features.to_frame().T
        else:
            frame = pd.DataFrame([features])
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return frame[self.feature_columns]

    def predict_proba(self, features: dict[str, Any] | pd.Series | pd.DataFrame) -> float:
        frame = self.as_frame(features)
        return float(self.model.predict_proba(frame)[0, 1])

    def predict_batch(self, frame: pd.DataFrame) -> pd.Series:
        prepared = self.as_frame(frame)
        return pd.Series(
            self.model.predict_proba(prepared)[:, 1],
            index=prepared.index,
            name="risk",
        )

    @staticmethod
    def risk_band(probability: float) -> str:
        for upper, label in RISK_BANDS:
            if probability < upper:
                return label
        return "High Risk"
