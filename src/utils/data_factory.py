from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def generate_synthetic_diabetes_dataset(
    output_path: str | Path,
    n_samples: int = 768,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic diabetes-style dataset.

    The generated data is only for demo/testing workflows and should not be
    interpreted as clinical evidence.
    """
    rng = np.random.default_rng(seed=random_state)

    age = rng.integers(21, 81, size=n_samples)
    pregnancies = np.clip(rng.poisson(lam=np.maximum((age - 18) / 12, 0.2)), 0, 17)
    glucose = np.clip(rng.normal(loc=120, scale=28, size=n_samples), 55, 220)
    blood_pressure = np.clip(rng.normal(loc=72, scale=12, size=n_samples), 38, 122)
    skin_thickness = np.clip(rng.normal(loc=23, scale=10, size=n_samples), 0, 99)
    insulin = np.clip(rng.lognormal(mean=4.3, sigma=0.75, size=n_samples), 0, 860)
    bmi = np.clip(rng.normal(loc=32, scale=7, size=n_samples), 16, 67)
    dpf = np.clip(rng.gamma(shape=2.0, scale=0.23, size=n_samples), 0.05, 2.6)

    # A simple risk function to produce plausible class balance.
    risk_score = (
        (glucose - 118) / 18
        + (bmi - 30) / 6
        + (age - 35) / 22
        + 0.12 * pregnancies
        + 0.9 * dpf
        + rng.normal(0, 1.0, size=n_samples)
    )
    risk_probability = 1 / (1 + np.exp(-risk_score))
    outcome = rng.binomial(n=1, p=np.clip(risk_probability, 0.02, 0.98), size=n_samples)

    dataset = pd.DataFrame(
        {
            "Pregnancies": pregnancies.astype(int),
            "Glucose": np.round(glucose, 1),
            "BloodPressure": np.round(blood_pressure, 1),
            "SkinThickness": np.round(skin_thickness, 1),
            "Insulin": np.round(insulin, 1),
            "BMI": np.round(bmi, 1),
            "DiabetesPedigreeFunction": np.round(dpf, 3),
            "Age": age.astype(int),
            "Outcome": outcome.astype(int),
        }
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return dataset
