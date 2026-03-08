from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.utils.preprocess import normalize_feature_frame, split_features_target


class PreprocessTests(unittest.TestCase):
    def test_normalize_feature_frame_replaces_expected_zero_values(self) -> None:
        frame = pd.DataFrame(
            {
                "Glucose": [0, 130],
                "BloodPressure": [70, 0],
                "Age": [33, 41],
            }
        )

        normalized = normalize_feature_frame(frame)

        self.assertTrue(np.isnan(normalized.loc[0, "Glucose"]))
        self.assertTrue(np.isnan(normalized.loc[1, "BloodPressure"]))
        self.assertEqual(normalized.loc[0, "Age"], 33)

    def test_split_features_target_raises_when_target_missing(self) -> None:
        frame = pd.DataFrame({"Glucose": [100, 120], "BMI": [28.5, 31.2]})
        with self.assertRaises(ValueError):
            split_features_target(frame, target_col="Outcome")


if __name__ == "__main__":
    unittest.main()
