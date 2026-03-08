from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Expected a CSV like data/diabetes.csv."
        )
    return pd.read_csv(path)


def normalize_feature_frame(
    frame: pd.DataFrame, zero_as_missing: Iterable[str] = DEFAULT_ZERO_AS_MISSING
) -> pd.DataFrame:
    normalized = frame.copy()
    target_cols = [col for col in zero_as_missing if col in normalized.columns]
    if target_cols:
        normalized[target_cols] = normalized[target_cols].replace(0, np.nan)
    return normalized


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist in dataset.")
    features = df.drop(columns=[target_col])
    target = df[target_col].astype(int)
    return features, target
