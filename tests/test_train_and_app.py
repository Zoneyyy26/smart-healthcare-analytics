from __future__ import annotations

import io
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
        self.users_path = self.artifact_root / "users_test.csv"
        self.reports_path = self.artifact_root / "reports_test.csv"
        self.predictions_path = self.artifact_root / "predictions_test.csv"

        self.dataset_path.unlink(missing_ok=True)
        self.model_path.unlink(missing_ok=True)
        self.users_path.unlink(missing_ok=True)
        self.reports_path.unlink(missing_ok=True)
        self.predictions_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.dataset_path.unlink(missing_ok=True)
        self.model_path.unlink(missing_ok=True)
        self.users_path.unlink(missing_ok=True)
        self.reports_path.unlink(missing_ok=True)
        self.predictions_path.unlink(missing_ok=True)

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

        with patch.dict(
            os.environ,
            {
                "MODEL_PATH": str(self.model_path),
                "USERS_PATH": str(self.users_path),
                "REPORTS_PATH": str(self.reports_path),
                "PREDICTIONS_PATH": str(self.predictions_path),
            },
        ):
            from src.app import create_app

            app = create_app()
            client = app.test_client()

        health_response = client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        health_payload = json.loads(health_response.data)
        self.assertEqual(health_payload["status"], "ok")

        docs_response = client.get("/doc")
        self.assertEqual(docs_response.status_code, 200)
        self.assertIn("Backend API Docs", docs_response.get_data(as_text=True))

        payload = {feature: 1.0 for feature in artifact["feature_names"]}
        unauth_response = client.post("/predict", json=payload)
        self.assertEqual(unauth_response.status_code, 401)

        with client.session_transaction() as session:
            session["logged_in"] = True
            session["username"] = "test-user"

        predict_response = client.post("/predict", json=payload)
        self.assertEqual(predict_response.status_code, 200)
        predict_payload = json.loads(predict_response.data)
        self.assertIn("prediction", predict_payload)
        self.assertIn("risk_probability", predict_payload)
        self.assertEqual(predict_payload["prediction_count"], 1)

        second_payload = dict(payload)
        second_payload[artifact["feature_names"][0]] = 2.0
        second_response = client.post("/predict", json=second_payload)
        self.assertEqual(second_response.status_code, 200)

        summary_response = client.get("/api/predictions/summary")
        self.assertEqual(summary_response.status_code, 200)
        summary_payload = json.loads(summary_response.data)
        self.assertEqual(summary_payload["count"], 2)
        self.assertIsNotNone(summary_payload["latest"])
        self.assertIsNotNone(summary_payload["previous"])

        report_payload = {feature: "1.0" for feature in artifact["feature_names"]}
        report_payload.update(
            {
                "patient_name": "Arjun Sharma",
                "patient_id": "PAT-1001",
                "patient_gender": "male",
                "patient_contact": "9999999999",
                "patient_notes": "Frequent thirst and fatigue",
                "report_file": (io.BytesIO(b"high glucose and high hba1c with polyuria"), "report.txt"),
            }
        )
        report_analysis_response = client.post(
            "/api/analyze-report",
            data=report_payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(report_analysis_response.status_code, 200)
        report_analysis_payload = json.loads(report_analysis_response.data)
        self.assertEqual(report_analysis_payload["status"], "success")
        self.assertIn("analysis", report_analysis_payload)
        self.assertIn("analysis_level", report_analysis_payload["analysis"])
        self.assertTrue(report_analysis_payload["saved_to_history"])

        incomplete = dict(payload)
        incomplete.pop(artifact["feature_names"][0])
        missing_response = client.post("/predict", json=incomplete)
        self.assertEqual(missing_response.status_code, 400)

        interactions_response = client.get("/api/interactions")
        self.assertEqual(interactions_response.status_code, 200)
        interactions_payload = json.loads(interactions_response.data)
        self.assertGreaterEqual(interactions_payload["count"], 4)
        self.assertTrue(
            any(entry.get("event_type") == "predict" for entry in interactions_payload["data"])
        )
        self.assertTrue(
            any(entry.get("event_type") == "analyze_report" for entry in interactions_payload["data"])
        )

        interaction_summary_response = client.get("/api/interactions/summary")
        self.assertEqual(interaction_summary_response.status_code, 200)
        interaction_summary_payload = json.loads(interaction_summary_response.data)
        self.assertIn("predict", interaction_summary_payload["by_event"])


if __name__ == "__main__":
    unittest.main()
