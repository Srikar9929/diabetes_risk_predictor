"""Personalized, non-prescriptive suggestions from model inputs and SHAP ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lifestyle_simulator import InterventionResult, LifestyleSimulator
from predictor import RiskPredictor


@dataclass
class PlanItem:
    priority: int
    feature: str
    title: str
    action: str
    current: str
    target: str
    new_value: float
    estimated_risk_drop_pp: float
    estimated_relative_drop: float
    tip: str


@dataclass
class PersonalPlan:
    baseline_risk: float
    projected_risk: float
    total_drop_pp: float
    band: str
    projected_band: str
    items: list[PlanItem]
    summary: str
    recommendations: list[str]


class PlanBuilder:
    def __init__(
        self,
        predictor: RiskPredictor | None = None,
        simulator: LifestyleSimulator | None = None,
    ) -> None:
        self.predictor = predictor or RiskPredictor()
        self.simulator = simulator or LifestyleSimulator(self.predictor)

    def build(self, features: dict[str, Any], top_k: int = 4) -> PersonalPlan:
        baseline = self.predictor.predict_proba(features)
        band = self.predictor.risk_band(baseline)
        interventions = self.simulator.recommend_targets(features, max_items=top_k)

        changes = {r.feature: r.new_value for r in interventions}
        _, projected, _ = self.simulator.what_if_combo(features, changes)
        projected_band = self.predictor.risk_band(projected)

        items: list[PlanItem] = []
        cumulative = dict(features)
        running_risk = baseline
        for i, r in enumerate(interventions, start=1):
            step = self.simulator.simulate_one(
                cumulative, r.feature, r.new_value, baseline_risk=running_risk
            )
            items.append(
                PlanItem(
                    priority=i,
                    feature=r.feature,
                    title=r.label,
                    action=_action_text(r),
                    current=_fmt(r.feature, r.baseline_value),
                    target=_fmt(r.feature, r.new_value),
                    new_value=float(step.new_value),
                    estimated_risk_drop_pp=round(step.risk_points, 1),
                    estimated_relative_drop=round(step.relative_drop * 100, 1),
                    tip=r.tip,
                )
            )
            cumulative = {**cumulative, r.feature: step.new_value}
            running_risk = step.new_risk

        total_pp = (baseline - projected) * 100
        if not items:
            summary = (
                f"The model currently estimates a {baseline:.0%} probability "
                f"({band}). Lifestyle-style inputs already look relatively favorable "
                "in this hypothetical profile. That is not a medical clearance."
            )
        else:
            summary = (
                f"The model currently estimates a {baseline:.0%} probability ({band}). "
                f"According to the ML model, this hypothetical combined input scenario "
                f"produces a predicted probability of about {projected:.0%} ({projected_band}), "
                f"a change of {total_pp:.1f} percentage points on the model's score. "
                "This is not a forecast of how much anyone's medical risk would actually change."
            )

        return PersonalPlan(
            baseline_risk=baseline,
            projected_risk=projected,
            total_drop_pp=total_pp,
            band=band,
            projected_band=projected_band,
            items=items,
            summary=summary,
            recommendations=build_recommendations(features),
        )


def build_recommendations(features: dict[str, Any]) -> list[str]:
    """Conservative, non-diagnostic suggestions tied to the entered values."""
    notes: list[str] = []
    sleep = float(features["sleep_hours"])
    steps = float(features["daily_steps"])
    active = float(features["active_minutes"])
    bmi = float(features["bmi"])
    glucose = float(features["glucose"])
    rhr = float(features["resting_hr"])
    hrv = float(features["hrv_ms"])
    bp = float(features["blood_pressure"])

    if sleep < 7:
        notes.append(
            "Your entered sleep duration is below a commonly discussed adult range of about "
            "7–9 hours. Consider discussing sleep habits with a qualified professional. "
            "This app cannot diagnose a sleep disorder."
        )
    if steps < 5000:
        notes.append(
            "Entered daily steps are relatively low. If appropriate for you, gradually "
            "increasing regular movement is a typical lifestyle topic to review with a clinician or physiotherapist."
        )
    elif steps < 8000:
        notes.append(
            "Entered daily steps are in a moderate range. Any activity change should be "
            "individual and medically appropriate."
        )
    if active < 20:
        notes.append(
            "Entered active minutes are relatively low. Consider asking a qualified professional "
            "what type and amount of activity is suitable for you."
        )
    if bmi >= 30:
        notes.append(
            "BMI is contributing as a high value in this profile. BMI is only a screening number. "
            "Discuss healthy weight-management options with a healthcare professional rather than following app targets."
        )
    elif bmi >= 25:
        notes.append(
            "BMI is in a range often labeled overweight in population charts. That label is not a diagnosis. "
            "A clinician can interpret it alongside your overall health."
        )
    if glucose >= 126:
        notes.append(
            "Entered fasting glucose is in a range that clinicians often investigate further. "
            "This is not a diagnosis of diabetes. Please seek in-person medical advice for lab interpretation."
        )
    elif glucose >= 100:
        notes.append(
            "Entered fasting glucose is above common fasting reference ranges used in screening literature. "
            "Only a licensed clinician can interpret a real lab result."
        )
    if rhr >= 90:
        notes.append(
            "Entered resting heart rate is relatively high. Heart rate has many causes. "
            "Do not self-treat based on this number."
        )
    if hrv < 30:
        notes.append(
            "Entered HRV is relatively low compared with the synthetic training distribution. "
            "Consumer HRV readings are noisy and are not a medical assessment."
        )
    if bp >= 90:
        notes.append(
            "Entered diastolic blood pressure is relatively high. Blood pressure must be measured "
            "properly and interpreted by a clinician."
        )

    notes.append(
        "None of these notes is a prescription, medication recommendation, or treatment plan. "
        "If you have symptoms or abnormal lab results, contact a licensed healthcare professional."
    )
    return notes


def _fmt(feature: str, value: float) -> str:
    if feature == "daily_steps":
        return f"{int(value):,} steps"
    if feature in {"active_minutes", "resting_hr"}:
        return f"{value:.0f}"
    if feature == "bmi":
        return f"{value:.1f}"
    if feature in {"sleep_hours", "hrv_ms"}:
        return f"{value:.1f}"
    return str(value)


def _action_text(r: InterventionResult) -> str:
    if r.new_value > r.baseline_value:
        return (
            f"Hypothetical model input: increase {r.label.lower()} from "
            f"{_fmt(r.feature, r.baseline_value)} to {_fmt(r.feature, r.new_value)}"
        )
    return (
        f"Hypothetical model input: reduce {r.label.lower()} from "
        f"{_fmt(r.feature, r.baseline_value)} to {_fmt(r.feature, r.new_value)}"
    )
