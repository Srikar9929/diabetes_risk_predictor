"""Generate a SYNTHETIC fused clinical + lifestyle diabetes dataset.

This is demonstration data for an educational ML project. It is NOT real
patient records. Clinical column names follow the well-known Pima-style
fields so the app can later swap in a real clinical table; wearable-style
fields are simulated so lifestyle what-if analysis is possible.

Do not treat rows as medically measured observations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLINICAL_FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]

LIFESTYLE_FEATURES = [
    "sleep_hours",
    "resting_hr",
    "daily_steps",
    "active_minutes",
    "hrv_ms",
]

FEATURE_COLUMNS = CLINICAL_FEATURES + LIFESTYLE_FEATURES
TARGET = "diabetes"

# Columns that are historically 0 = "missing" in the original Pima table.
# On this synthetic set we do NOT recode zeros unless a profile says so.
PIMA_ZERO_AS_MISSING = [
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
]


def _clip(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(arr, low, high)


def generate_fused_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create correlated synthetic hospital + lifestyle records and a label.

    A latent metabolic factor drives both clinical markers and lifestyle
    signals so the table is internally consistent. The binary label is then
    sampled from a logistic function of those features (plus noise).
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 75, size=n_samples)
    pregnancies = np.where(
        age < 45,
        rng.integers(0, 6, size=n_samples),
        rng.integers(0, 10, size=n_samples),
    )

    metabolic = rng.normal(0, 1, size=n_samples)

    glucose = _clip(110 + 28 * metabolic + rng.normal(0, 12, n_samples), 70, 220)
    bmi = _clip(28 + 5.5 * metabolic + rng.normal(0, 3.5, n_samples), 18, 55)
    blood_pressure = _clip(72 + 10 * metabolic + rng.normal(0, 8, n_samples), 40, 122)
    skin_thickness = _clip(25 + 8 * metabolic + rng.normal(0, 6, n_samples), 0, 80)
    insulin = _clip(90 + 70 * metabolic + rng.normal(0, 40, n_samples), 0, 600)
    diabetes_pedigree = _clip(
        0.45 + 0.25 * metabolic + rng.normal(0, 0.2, n_samples), 0.08, 2.5
    )

    sleep_hours = _clip(7.2 - 0.55 * metabolic + rng.normal(0, 0.9, n_samples), 3.5, 10)
    resting_hr = _clip(72 + 8 * metabolic + rng.normal(0, 6, n_samples), 48, 110)
    daily_steps = _clip(
        8500 - 1800 * metabolic + rng.normal(0, 1200, n_samples), 800, 18000
    ).astype(int)
    active_minutes = _clip(
        45 - 12 * metabolic + rng.normal(0, 10, n_samples), 0, 150
    ).astype(int)
    hrv_ms = _clip(55 - 10 * metabolic + rng.normal(0, 8, n_samples), 15, 100)

    # Prevalence is tuned toward a minority positive class (~30%), not 50/50.
    logit = (
        -3.4
        + 0.038 * (glucose - 100)
        + 0.10 * (bmi - 25)
        + 0.028 * (age - 40)
        + 1.2 * diabetes_pedigree
        + 0.05 * pregnancies
        + 0.014 * (blood_pressure - 70)
        + 0.0018 * insulin
        - 0.42 * (sleep_hours - 7)
        + 0.032 * (resting_hr - 70)
        - 0.00015 * (daily_steps - 8000)
        - 0.022 * (active_minutes - 40)
        - 0.025 * (hrv_ms - 50)
        + rng.normal(0, 0.45, n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    diabetes = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "pregnancies": pregnancies,
            "glucose": np.round(glucose, 1),
            "blood_pressure": np.round(blood_pressure, 1),
            "skin_thickness": np.round(skin_thickness, 1),
            "insulin": np.round(insulin, 1),
            "bmi": np.round(bmi, 1),
            "diabetes_pedigree": np.round(diabetes_pedigree, 3),
            "age": age,
            "sleep_hours": np.round(sleep_hours, 1),
            "resting_hr": np.round(resting_hr, 1),
            "daily_steps": daily_steps,
            "active_minutes": active_minutes,
            "hrv_ms": np.round(hrv_ms, 1),
            TARGET: diabetes,
        }
    )


def save_dataset(
    path: Path | str | None = None, n_samples: int = 2500, seed: int = 42
) -> Path:
    root = Path(__file__).resolve().parent
    out = Path(path) if path else root / "data" / "diabetes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_fused_dataset(n_samples=n_samples, seed=seed)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    dest = save_dataset()
    df = pd.read_csv(dest)
    print(f"Wrote {dest} ({len(df)} rows, prevalence={df[TARGET].mean():.1%})")
    print("NOTE: this file is SYNTHETIC demonstration data, not clinical records.")
