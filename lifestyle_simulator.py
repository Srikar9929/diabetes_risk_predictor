"""Counterfactual lifestyle simulation using the trained model (not medical effects)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from generate_dataset import FEATURE_COLUMNS
from predictor import RiskPredictor

LIFESTYLE_LEVERS = {
    "sleep_hours": {
        "label": "Sleep hours / night",
        "unit": "hrs",
        "direction": "increase",
        "step": 0.5,
        "min": 4.0,
        "max": 9.5,
        "tip": "Many adults aim for a consistent 7–9 hours of sleep. Discuss sleep concerns with a qualified professional if they persist.",
    },
    "daily_steps": {
        "label": "Daily steps",
        "unit": "steps",
        "direction": "increase",
        "step": 1000,
        "min": 2000,
        "max": 15000,
        "tip": "If it is appropriate for you, gradually increasing daily movement is a common lifestyle goal. A clinician can advise what is safe.",
    },
    "active_minutes": {
        "label": "Active minutes / day",
        "unit": "min",
        "direction": "increase",
        "step": 10,
        "min": 0,
        "max": 120,
        "tip": "Public health guidance often discusses regular moderate activity. Individual plans should be checked with a professional.",
    },
    "resting_hr": {
        "label": "Resting heart rate",
        "unit": "bpm",
        "direction": "decrease",
        "step": 2,
        "min": 50,
        "max": 100,
        "tip": "Resting heart rate is influenced by fitness, stress, sleep, and medical conditions. Do not treat a slider change as a treatment plan.",
    },
    "hrv_ms": {
        "label": "Heart-rate variability",
        "unit": "ms",
        "direction": "increase",
        "step": 5,
        "min": 20,
        "max": 90,
        "tip": "HRV estimates vary by device. Use this only as a hypothetical model input, not a diagnosis of recovery or stress.",
    },
    "bmi": {
        "label": "BMI",
        "unit": "",
        "direction": "decrease",
        "step": 0.5,
        "min": 18.5,
        "max": 45,
        "tip": "BMI is a crude screening number. Weight-management decisions belong with a healthcare professional.",
    },
}


@dataclass
class InterventionResult:
    feature: str
    label: str
    baseline_value: float
    new_value: float
    baseline_risk: float
    new_risk: float
    absolute_drop: float
    relative_drop: float
    tip: str

    @property
    def risk_points(self) -> float:
        """Percentage-point change in the model's probability (positive = predicted risk went down)."""
        return self.absolute_drop * 100


class LifestyleSimulator:
    def __init__(self, predictor: RiskPredictor | None = None) -> None:
        self.predictor = predictor or RiskPredictor()

    def _apply(self, base: dict[str, Any], feature: str, new_value: float) -> dict[str, Any]:
        updated = dict(base)
        meta = LIFESTYLE_LEVERS[feature]
        lo, hi = meta["min"], meta["max"]
        updated[feature] = float(max(lo, min(hi, new_value)))
        return updated

    def simulate_one(
        self,
        features: dict[str, Any],
        feature: str,
        new_value: float,
        baseline_risk: float | None = None,
    ) -> InterventionResult:
        if feature not in LIFESTYLE_LEVERS:
            raise KeyError(f"Unknown lifestyle feature: {feature}")
        base_risk = (
            baseline_risk
            if baseline_risk is not None
            else self.predictor.predict_proba(features)
        )
        updated = self._apply(features, feature, new_value)
        new_risk = self.predictor.predict_proba(updated)
        abs_drop = base_risk - new_risk
        rel = abs_drop / base_risk if base_risk > 1e-6 else 0.0
        meta = LIFESTYLE_LEVERS[feature]
        return InterventionResult(
            feature=feature,
            label=meta["label"],
            baseline_value=float(features[feature]),
            new_value=float(updated[feature]),
            baseline_risk=base_risk,
            new_risk=new_risk,
            absolute_drop=abs_drop,
            relative_drop=rel,
            tip=meta["tip"],
        )

    def recommend_targets(
        self, features: dict[str, Any], max_items: int = 5
    ) -> list[InterventionResult]:
        """Rank hypothetical targets by how much they change the *model* probability."""
        baseline = self.predictor.predict_proba(features)
        results: list[InterventionResult] = []

        targets = {
            "sleep_hours": max(float(features["sleep_hours"]), 7.5),
            "daily_steps": max(float(features["daily_steps"]), 10000),
            "active_minutes": max(float(features["active_minutes"]), 45),
            "resting_hr": min(float(features["resting_hr"]), 65),
            "hrv_ms": max(float(features["hrv_ms"]), 55),
            "bmi": min(float(features["bmi"]), max(24.0, float(features["bmi"]) - 2)),
        }

        for feature, target in targets.items():
            current = float(features[feature])
            if abs(target - current) < 1e-6:
                continue
            result = self.simulate_one(features, feature, target, baseline_risk=baseline)
            if result.absolute_drop > 0.005:
                results.append(result)

        results.sort(key=lambda r: r.absolute_drop, reverse=True)
        return results[:max_items]

    def what_if_combo(
        self, features: dict[str, Any], changes: dict[str, float]
    ) -> tuple[float, float, dict[str, Any]]:
        baseline = self.predictor.predict_proba(features)
        updated = dict(features)
        for feature, value in changes.items():
            if feature in LIFESTYLE_LEVERS:
                updated = self._apply(updated, feature, value)
        new_risk = self.predictor.predict_proba(updated)
        return baseline, new_risk, updated

    def sensitivity_curve(
        self, features: dict[str, Any], feature: str, points: int = 12
    ) -> pd.DataFrame:
        meta = LIFESTYLE_LEVERS[feature]
        grid = [
            meta["min"] + i * (meta["max"] - meta["min"]) / (points - 1)
            for i in range(points)
        ]
        risks = [self.simulate_one(features, feature, float(val)).new_risk for val in grid]
        return pd.DataFrame({"value": grid, "risk": risks, "feature": feature})


def features_from_inputs(**kwargs: Any) -> dict[str, Any]:
    return {col: kwargs[col] for col in FEATURE_COLUMNS}
