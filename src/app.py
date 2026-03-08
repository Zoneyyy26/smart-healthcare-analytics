from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

DEFAULT_MODEL_PATH = Path("models/diabetes_model.joblib")


def load_artifact(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. Run training first: python -m src.train"
        )
    artifact = joblib.load(model_path)
    required_keys = {"model", "feature_names"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    return artifact


def get_positive_probability(model, feature_frame: pd.DataFrame) -> float | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)
        return float(probabilities[0][1])

    if hasattr(model, "decision_function"):
        score = float(np.atleast_1d(model.decision_function(feature_frame))[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    return None


def create_app() -> Flask:
    app = Flask(__name__)
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    state = {"artifact": None, "error": None}

    try:
        state["artifact"] = load_artifact(model_path)
    except Exception as exc:  # noqa: BLE001
        state["error"] = str(exc)

    @app.get("/health")
    def health():
        if state["artifact"] is None:
            return (
                jsonify({"status": "error", "message": state["error"], "model_path": str(model_path)}),
                500,
            )
        return jsonify({"status": "ok", "model_path": str(model_path)})

    @app.post("/predict")
    def predict():
        if state["artifact"] is None:
            return jsonify({"error": state["error"]}), 500

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON body. Expected an object with feature values."}), 400

        feature_names = state["artifact"]["feature_names"]
        missing_features = [name for name in feature_names if name not in payload]
        if missing_features:
            return (
                jsonify(
                    {
                        "error": "Missing required features.",
                        "missing_features": missing_features,
                    }
                ),
                400,
            )

        try:
            row = {name: float(payload[name]) for name in feature_names}
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"All features must be numeric. {exc}"}), 400

        feature_frame = pd.DataFrame([row], columns=feature_names)
        model = state["artifact"]["model"]

        prediction = int(model.predict(feature_frame)[0])
        probability = get_positive_probability(model, feature_frame)

        response = {"prediction": prediction, "risk_probability": probability}
        if probability is None:
            response["note"] = "This model does not expose calibrated probability scores."
        return jsonify(response)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
