from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib


class TrainAndApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.artifact_root = self.repo_root / "tests" / ".artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.dataset_path = self.artifact_root / "diabetes_test_dataset.csv"
        self.model_path = self.artifact_root / "diabetes_test_model.joblib"

        self.dataset_path.unlink(missing_ok=True)
        self.model_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.dataset_path.unlink(missing_ok=True)
        self.model_path.unlink(missing_ok=True)

    def _run_training_cli_in_process(self) -> None:
        from src.train import main as train_main

        argv = [
            "train.py",
            "--dataset-path",
            str(self.dataset_path),
            "--model-output",
            str(self.model_path),
            "--synthetic-samples",
            "240",
            "--random-state",
            "21",
        ]

        with patch.object(sys, "argv", argv):
            train_main()

    def test_train_cli_generates_dataset_and_artifact(self) -> None:
        self._run_training_cli_in_process()

        self.assertTrue(self.dataset_path.exists())
        self.assertTrue(self.model_path.exists())

        artifact = joblib.load(self.model_path)
        self.assertIn("model", artifact)
        self.assertIn("feature_names", artifact)
        self.assertIn("metrics", artifact)
        self.assertIn("selected_model", artifact)

    def test_app_health_and_predict(self) -> None:
        self._run_training_cli_in_process()
        artifact = joblib.load(self.model_path)

        with patch.dict(os.environ, {"MODEL_PATH": str(self.model_path)}):
            from src.app import create_app

            app = create_app()
            client = app.test_client()

        health_response = client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        health_payload = json.loads(health_response.data)
        self.assertEqual(health_payload["status"], "ok")

        payload = {feature: 1.0 for feature in artifact["feature_names"]}
        predict_response = client.post("/predict", json=payload)
        self.assertEqual(predict_response.status_code, 200)
        predict_payload = json.loads(predict_response.data)
        self.assertIn("prediction", predict_payload)
        self.assertIn("risk_probability", predict_payload)

        incomplete = dict(payload)
        incomplete.pop(artifact["feature_names"][0])
        missing_response = client.post("/predict", json=incomplete)
        self.assertEqual(missing_response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
