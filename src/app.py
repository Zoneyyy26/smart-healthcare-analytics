from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Add project root to sys.path to allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from functools import wraps

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - fallback for minimal test environments
    def CORS(_app):  # type: ignore[no-redef]
        return None

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
    app.secret_key = os.getenv("SECRET_KEY", "smarthealthcare-secret-2026")
    CORS(app)
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    state = {"artifact": None, "error": None}

    import csv as _csv
    import contextlib as _contextlib
    import datetime as _dt
    import hashlib
    import io as _io
    import json as _json
    import math as _math
    import sqlite3 as _sqlite3

    USERS_PATH = Path(os.getenv("USERS_PATH", "data/users.csv"))  # legacy CSV migration source
    REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "data/reports.csv"))  # legacy CSV migration source
    PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_PATH", "data/predictions.csv"))  # legacy CSV migration source
    explicit_db_path = os.getenv("APP_DB_PATH")
    env_predictions_path = os.getenv("PREDICTIONS_PATH")
    if explicit_db_path:
        DB_PATH = Path(explicit_db_path)
    elif env_predictions_path and env_predictions_path != "data/predictions.csv":
        # Keeps test isolation by honoring custom PREDICTIONS_PATH env values.
        DB_PATH = Path(env_predictions_path)
    else:
        DB_PATH = Path("data/user_interactions.db")

    MAX_REPORT_UPLOAD_BYTES = 2 * 1024 * 1024
    ALLOWED_REPORT_EXTENSIONS = {".txt", ".csv", ".json", ".md", ".log", ".tsv", ".xml"}
    PREDICTION_FEATURE_COLUMNS = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _hash(pwd: str) -> str:
        return hashlib.sha256(pwd.encode()).hexdigest()

    def _to_sql_float(value: object) -> float | None:
        parsed = _safe_float(value)
        return float(parsed) if parsed is not None else None

    def _open_db() -> _sqlite3.Connection:
        _ensure_parent(DB_PATH)
        conn = _sqlite3.connect(DB_PATH)
        conn.row_factory = _sqlite3.Row
        return conn

    @_contextlib.contextmanager
    def _db_conn():
        conn = _open_db()
        try:
            yield conn
        finally:
            conn.close()

    def _load_csv_rows(path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            with open(path, newline="", encoding="utf-8") as file:
                return list(_csv.DictReader(file))
        except Exception:
            return []

    def _migrate_legacy_csv_if_needed() -> None:
        try:
            db_resolved = DB_PATH.resolve()
        except Exception:
            db_resolved = DB_PATH

        with _db_conn() as conn:
            user_count = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            if user_count == 0 and USERS_PATH.exists():
                try:
                    users_resolved = USERS_PATH.resolve()
                except Exception:
                    users_resolved = USERS_PATH
                if users_resolved != db_resolved:
                    for row in _load_csv_rows(USERS_PATH):
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO users
                            (username, password_hash, first_name, last_name, email, role, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(row.get("username", "")).strip(),
                                str(row.get("password_hash", "")),
                                str(row.get("first_name", "")),
                                str(row.get("last_name", "")),
                                str(row.get("email", "")),
                                str(row.get("role", "patient") or "patient"),
                                str(row.get("created_at", "")),
                            ),
                        )

            prediction_count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            if prediction_count == 0 and PREDICTIONS_PATH.exists():
                try:
                    predictions_resolved = PREDICTIONS_PATH.resolve()
                except Exception:
                    predictions_resolved = PREDICTIONS_PATH
                if predictions_resolved != db_resolved:
                    for row in _load_csv_rows(PREDICTIONS_PATH):
                        conn.execute(
                            """
                            INSERT INTO predictions (
                                username, submitted_at, prediction, risk_probability, patient_name, patient_id,
                                patient_gender, patient_contact, patient_notes, report_file_name,
                                report_analysis_level, Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(row.get("username", "")),
                                str(row.get("submitted_at", "")),
                                _safe_int(row.get("prediction")),
                                _to_sql_float(row.get("risk_probability")),
                                str(row.get("patient_name", "")),
                                str(row.get("patient_id", "")),
                                str(row.get("patient_gender", "")),
                                str(row.get("patient_contact", "")),
                                str(row.get("patient_notes", "")),
                                str(row.get("report_file_name", "")),
                                str(row.get("report_analysis_level", "")),
                                _to_sql_float(row.get("Pregnancies")),
                                _to_sql_float(row.get("Glucose")),
                                _to_sql_float(row.get("BloodPressure")),
                                _to_sql_float(row.get("SkinThickness")),
                                _to_sql_float(row.get("Insulin")),
                                _to_sql_float(row.get("BMI")),
                                _to_sql_float(row.get("DiabetesPedigreeFunction")),
                                _to_sql_float(row.get("Age")),
                            ),
                        )

            report_count = int(conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0])
            if report_count == 0 and REPORTS_PATH.exists():
                try:
                    reports_resolved = REPORTS_PATH.resolve()
                except Exception:
                    reports_resolved = REPORTS_PATH
                if reports_resolved != db_resolved:
                    for row in _load_csv_rows(REPORTS_PATH):
                        conn.execute(
                            """
                            INSERT INTO reports (
                                submitted_by, submitted_at, patient_name, age, Pregnancies, Glucose,
                                BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, notes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(row.get("submitted_by", "")),
                                str(row.get("submitted_at", "")),
                                str(row.get("patient_name", "")),
                                _safe_int(row.get("age")),
                                _to_sql_float(row.get("Pregnancies")),
                                _to_sql_float(row.get("Glucose")),
                                _to_sql_float(row.get("BloodPressure")),
                                _to_sql_float(row.get("SkinThickness")),
                                _to_sql_float(row.get("Insulin")),
                                _to_sql_float(row.get("BMI")),
                                _to_sql_float(row.get("DiabetesPedigreeFunction")),
                                str(row.get("notes", "")),
                            ),
                        )

            conn.commit()

    def _init_database() -> None:
        with _db_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    email TEXT,
                    role TEXT NOT NULL DEFAULT 'patient',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    prediction INTEGER,
                    risk_probability REAL,
                    patient_name TEXT,
                    patient_id TEXT,
                    patient_gender TEXT,
                    patient_contact TEXT,
                    patient_notes TEXT,
                    report_file_name TEXT,
                    report_analysis_level TEXT,
                    Pregnancies REAL,
                    Glucose REAL,
                    BloodPressure REAL,
                    SkinThickness REAL,
                    Insulin REAL,
                    BMI REAL,
                    DiabetesPedigreeFunction REAL,
                    Age REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submitted_by TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    age INTEGER,
                    Pregnancies REAL,
                    Glucose REAL,
                    BloodPressure REAL,
                    SkinThickness REAL,
                    Insulin REAL,
                    BMI REAL,
                    DiabetesPedigreeFunction REAL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_method TEXT,
                    request_path TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_interactions_username_created_at
                ON user_interactions (username, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_interactions_event_type_created_at
                ON user_interactions (event_type, created_at)
                """
            )
            conn.commit()

        _migrate_legacy_csv_if_needed()

    def _load_users() -> dict:
        users: dict = {}
        with _db_conn() as conn:
            rows = conn.execute(
                """
                SELECT username, password_hash, first_name, last_name, email, role, created_at
                FROM users
                """
            ).fetchall()
        for row in rows:
            users[str(row["username"])] = {
                "username": row["username"],
                "password_hash": row["password_hash"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "email": row["email"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
        return users

    def _save_user(row: dict) -> None:
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, first_name, last_name, email, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("username", "")).strip(),
                    str(row.get("password_hash", "")),
                    str(row.get("first_name", "")),
                    str(row.get("last_name", "")),
                    str(row.get("email", "")),
                    str(row.get("role", "patient") or "patient"),
                    str(row.get("created_at", "")),
                ),
            )
            conn.commit()

    # Hardcoded admin always valid
    ADMIN_USER = os.getenv("APP_USER", "admin")
    ADMIN_PASS = os.getenv("APP_PASS", "admin123")

    def _authenticate(uname: str, pwd: str) -> bool:
        if uname == ADMIN_USER and pwd == ADMIN_PASS:
            return True
        users = _load_users()
        user = users.get(uname)
        if user and user["password_hash"] == _hash(pwd):
            return True
        return False

    def _get_user_role(uname: str) -> str:
        if uname == ADMIN_USER:
            return "admin"
        with _db_conn() as conn:
            row = conn.execute("SELECT role FROM users WHERE username = ?", (uname,)).fetchone()
        if row and row["role"]:
            return str(row["role"])
        return "patient"

    def _safe_float(value: object) -> float | None:
        if value in (None, "", "None"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(value: object) -> int | None:
        if value in (None, "", "None"):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _clean_text(value: object, max_len: int = 300) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text[:max_len]

    def _log_user_interaction(
        event_type: str,
        status: str = "success",
        *,
        details: dict | None = None,
        username: str | None = None,
    ) -> None:
        try:
            resolved_username = username
            if resolved_username is None:
                resolved_username = str(session.get("username", "") or "")

            details_blob = ""
            if details:
                details_blob = _json.dumps(details, ensure_ascii=False)[:2000]

            with _db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO user_interactions (
                        username, event_type, status, request_method, request_path,
                        ip_address, user_agent, details, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _clean_text(resolved_username, max_len=80),
                        _clean_text(event_type, max_len=60),
                        _clean_text(status, max_len=20),
                        _clean_text(request.method, max_len=12),
                        _clean_text(request.path, max_len=160),
                        _clean_text(request.headers.get("X-Forwarded-For") or request.remote_addr or "", max_len=80),
                        _clean_text(request.headers.get("User-Agent", ""), max_len=250),
                        details_blob,
                        _dt.datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                conn.commit()
        except Exception:
            return

    def _extract_patient_details(source: dict) -> dict:
        raw_details = source.get("patient_details", source)
        if raw_details in (None, ""):
            raw_details = {}
        if not isinstance(raw_details, dict):
            raise ValueError("patient_details must be an object.")
        return {
            "patient_name": _clean_text(raw_details.get("patient_name", ""), max_len=120),
            "patient_id": _clean_text(raw_details.get("patient_id", ""), max_len=80),
            "patient_gender": _clean_text(raw_details.get("patient_gender", ""), max_len=30),
            "patient_contact": _clean_text(raw_details.get("patient_contact", ""), max_len=80),
            "patient_notes": _clean_text(raw_details.get("patient_notes", ""), max_len=400),
        }

    def _upsert_csv_row(path: Path, row: dict) -> None:
        _ensure_parent(path)
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        with open(path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []

        missing_fields = [key for key in row.keys() if key not in existing_fields]
        if missing_fields:
            merged_fields = existing_fields + missing_fields
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=merged_fields)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
                writer.writerow(row)
            return

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow(row)

    def _read_uploaded_report(file_storage) -> tuple[str, str]:
        filename = _clean_text(file_storage.filename or "", max_len=160)
        if not filename:
            raise ValueError("No file selected.")

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_REPORT_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_REPORT_EXTENSIONS))
            raise ValueError(f"Unsupported report type. Allowed: {allowed}")

        content = file_storage.read(MAX_REPORT_UPLOAD_BYTES + 1)
        if not content:
            raise ValueError("Uploaded report is empty.")
        if len(content) > MAX_REPORT_UPLOAD_BYTES:
            raise ValueError("Report file is too large. Max allowed size is 2 MB.")

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")

        text = text.strip()
        if not text:
            raise ValueError("Uploaded report does not contain readable text.")
        return text, filename

    def _analyze_report_text(report_text: str) -> dict:
        lowered = report_text.lower()
        high_risk_terms = [
            "hyperglycemia",
            "high glucose",
            "elevated glucose",
            "high hba1c",
            "uncontrolled diabetes",
            "polyuria",
            "polydipsia",
            "ketoacidosis",
            "diabetic neuropathy",
        ]
        lower_risk_terms = [
            "normal glucose",
            "normoglycemia",
            "controlled hba1c",
            "no diabetes",
            "stable sugars",
        ]

        high_hits = [term for term in high_risk_terms if term in lowered]
        low_hits = [term for term in lower_risk_terms if term in lowered]
        score = len(high_hits) - len(low_hits)

        if score >= 2:
            level = "high"
        elif score == 1:
            level = "moderate"
        elif score <= -1:
            level = "low"
        else:
            level = "unclear"

        patterns = {
            "glucose": r"(?:glucose|fasting glucose)\s*[:=]?\s*(\d+(?:\.\d+)?)",
            "hba1c": r"(?:hba1c|a1c)\s*[:=]?\s*(\d+(?:\.\d+)?)",
            "bmi": r"(?:bmi|body mass index)\s*[:=]?\s*(\d+(?:\.\d+)?)",
            "blood_pressure": r"(?:blood pressure|bp)\s*[:=]?\s*(\d{2,3}(?:/\d{2,3})?)",
        }
        extracted_metrics: dict[str, str | float] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, lowered)
            if match:
                value = match.group(1)
                try:
                    extracted_metrics[key] = float(value)
                except ValueError:
                    extracted_metrics[key] = value

        compact = " ".join(report_text.split())
        preview = compact[:240] + ("..." if len(compact) > 240 else "")
        summary = (
            f"Report analysis indicates {level} risk signals "
            f"({len(high_hits)} high-risk cues, {len(low_hits)} low-risk cues)."
        )
        return {
            "analysis_level": level,
            "summary": summary,
            "evidence": sorted(set(high_hits + low_hits))[:10],
            "extracted_metrics": extracted_metrics,
            "preview": preview,
        }

    def _prediction_snapshot(record: dict | None) -> dict | None:
        if not record:
            return None
        return {
            "prediction": record.get("prediction"),
            "risk_probability": record.get("risk_probability"),
            "submitted_at": record.get("submitted_at"),
        }

    def _build_prediction_summary(predictions: list[dict]) -> dict:
        latest = predictions[0] if predictions else None
        previous = predictions[1] if len(predictions) > 1 else None
        delta_probability = None
        trend = "not_available"

        if (
            latest is not None
            and previous is not None
            and latest.get("risk_probability") is not None
            and previous.get("risk_probability") is not None
        ):
            delta_probability = float(latest["risk_probability"] - previous["risk_probability"])
            if delta_probability > 0.005:
                trend = "up"
            elif delta_probability < -0.005:
                trend = "down"
            else:
                trend = "steady"

        return {
            "count": len(predictions),
            "latest": _prediction_snapshot(latest),
            "previous": _prediction_snapshot(previous),
            "delta_probability": delta_probability,
            "trend": trend,
        }

    def _format_prediction_label(value: object) -> str:
        return "Diabetic" if _safe_int(value) == 1 else "Healthy"

    def _format_probability(value: object) -> str:
        parsed = _safe_float(value)
        if parsed is None:
            return "N/A"
        return f"{parsed * 100:.2f}%"

    def _build_prediction_text_report(record: dict) -> str:
        lines = [
            "Smart Healthcare Analytics - Prediction Report",
            "=" * 48,
            f"Submitted At: {record.get('submitted_at', 'N/A')}",
            f"Patient Name: {record.get('patient_name', 'N/A') or 'N/A'}",
            f"Patient ID: {record.get('patient_id', 'N/A') or 'N/A'}",
            f"Prediction: {_format_prediction_label(record.get('prediction'))}",
            f"Risk Probability: {_format_probability(record.get('risk_probability'))}",
            "",
            "Clinical Inputs",
            "-" * 48,
            f"Pregnancies: {record.get('Pregnancies', 'N/A')}",
            f"Glucose: {record.get('Glucose', 'N/A')}",
            f"Blood Pressure: {record.get('BloodPressure', 'N/A')}",
            f"Skin Thickness: {record.get('SkinThickness', 'N/A')}",
            f"Insulin: {record.get('Insulin', 'N/A')}",
            f"BMI: {record.get('BMI', 'N/A')}",
            f"Diabetes Pedigree Function: {record.get('DiabetesPedigreeFunction', 'N/A')}",
            f"Age: {record.get('Age', 'N/A')}",
            "",
            "Context",
            "-" * 48,
            f"Report File: {record.get('report_file_name', 'N/A') or 'N/A'}",
            f"Report Analysis Level: {record.get('report_analysis_level', 'N/A') or 'N/A'}",
            f"Notes: {record.get('patient_notes', '') or 'N/A'}",
        ]
        return "\n".join(lines)

    def _get_model_insights() -> dict:
        artifact = state.get("artifact")
        if not artifact:
            return {
                "selected_model": None,
                "metrics": {},
                "feature_importance": [],
            }

        model = artifact.get("model")
        feature_names = artifact.get("feature_names", [])
        metrics = artifact.get("metrics", {})
        selected_model = artifact.get("selected_model")
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("model", model)

        feature_importance: list[dict[str, float | str]] = []
        raw_scores: list[float] = []
        if hasattr(estimator, "feature_importances_"):
            raw_scores = [float(value) for value in estimator.feature_importances_]
        elif hasattr(estimator, "coef_"):
            raw_scores = [float(value) for value in np.abs(np.atleast_2d(estimator.coef_)[0])]

        total_score = float(sum(raw_scores))
        if raw_scores and total_score > 0:
            pairs = sorted(
                zip(feature_names, raw_scores, strict=False),
                key=lambda item: item[1],
                reverse=True,
            )
            feature_importance = [
                {
                    "feature": str(name),
                    "score": float(score),
                    "share": float(score / total_score),
                }
                for name, score in pairs
            ]

        return {
            "selected_model": selected_model,
            "metrics": metrics,
            "feature_importance": feature_importance,
        }

    def _save_prediction(
        username: str,
        feature_values: dict[str, float],
        prediction: int,
        probability: float | None,
        patient_details: dict | None = None,
        report_context: dict | None = None,
    ) -> None:
        patient_details = patient_details or {}
        report_context = report_context or {}
        with _db_conn() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    username, submitted_at, prediction, risk_probability,
                    patient_name, patient_id, patient_gender, patient_contact, patient_notes,
                    report_file_name, report_analysis_level,
                    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                    BMI, DiabetesPedigreeFunction, Age
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    _dt.datetime.now().isoformat(timespec="seconds"),
                    int(prediction),
                    probability,
                    _clean_text(patient_details.get("patient_name", ""), max_len=120),
                    _clean_text(patient_details.get("patient_id", ""), max_len=80),
                    _clean_text(patient_details.get("patient_gender", ""), max_len=30),
                    _clean_text(patient_details.get("patient_contact", ""), max_len=80),
                    _clean_text(patient_details.get("patient_notes", ""), max_len=400),
                    _clean_text(report_context.get("report_file_name", ""), max_len=160),
                    _clean_text(report_context.get("report_analysis_level", ""), max_len=30),
                    _to_sql_float(feature_values.get("Pregnancies")),
                    _to_sql_float(feature_values.get("Glucose")),
                    _to_sql_float(feature_values.get("BloodPressure")),
                    _to_sql_float(feature_values.get("SkinThickness")),
                    _to_sql_float(feature_values.get("Insulin")),
                    _to_sql_float(feature_values.get("BMI")),
                    _to_sql_float(feature_values.get("DiabetesPedigreeFunction")),
                    _to_sql_float(feature_values.get("Age")),
                ),
            )
            conn.commit()

    def _load_predictions_for_user(username: str) -> list[dict]:
        with _db_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    username, submitted_at, prediction, risk_probability,
                    patient_name, patient_id, patient_gender, patient_contact, patient_notes,
                    report_file_name, report_analysis_level,
                    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                    BMI, DiabetesPedigreeFunction, Age
                FROM predictions
                WHERE username = ?
                ORDER BY submitted_at DESC, id DESC
                """,
                (username,),
            ).fetchall()

        records: list[dict] = []
        for row in rows:
            parsed = {
                "username": row["username"],
                "submitted_at": row["submitted_at"],
                "prediction": _safe_int(row["prediction"]),
                "risk_probability": _safe_float(row["risk_probability"]),
                "patient_name": row["patient_name"] or "",
                "patient_id": row["patient_id"] or "",
                "patient_gender": row["patient_gender"] or "",
                "patient_contact": row["patient_contact"] or "",
                "patient_notes": row["patient_notes"] or "",
                "report_file_name": row["report_file_name"] or "",
                "report_analysis_level": row["report_analysis_level"] or "",
            }
            for feature_col in PREDICTION_FEATURE_COLUMNS:
                parsed[feature_col] = row[feature_col]
            records.append(parsed)
        return records

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    def api_login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("logged_in") or not session.get("username"):
                return jsonify({"error": "Authentication required."}), 401
            return f(*args, **kwargs)
        return decorated

    _init_database()

    try:
        state["artifact"] = load_artifact(model_path)
    except Exception as exc:  # noqa: BLE001
        state["error"] = str(exc)

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            username=session.get("username"),
            role=session.get("role"),
            logged_in=bool(session.get("logged_in")),
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("logged_in"):
            return redirect(url_for("home"))
        error = None
        if request.method == "POST":
            uname = request.form.get("username", "").strip()
            pwd   = request.form.get("password", "")
            if _authenticate(uname, pwd):
                session["logged_in"] = True
                session["username"]  = uname
                session["role"] = _get_user_role(uname)
                _log_user_interaction(
                    "login",
                    details={"role": session.get("role", "patient")},
                    username=uname,
                )
                return redirect(url_for("home"))
            error = "Invalid username or password. Please try again."
            _log_user_interaction("login", "failed", details={"reason": "invalid_credentials"}, username=uname)
        return render_template("login.html", error=error)

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if session.get("logged_in"):
            return redirect(url_for("home"))
        error = None
        success = None
        if request.method == "POST":
            first  = request.form.get("first_name", "").strip()
            last   = request.form.get("last_name", "").strip()
            email  = request.form.get("email", "").strip()
            uname  = request.form.get("username", "").strip()
            pwd    = request.form.get("password", "")
            cpwd   = request.form.get("confirm_password", "")
            role   = request.form.get("role", "patient")

            if not all([first, email, uname, pwd]):
                error = "Please fill in all required fields."
            elif len(pwd) < 6:
                error = "Password must be at least 6 characters."
            elif pwd != cpwd:
                error = "Passwords do not match."
            elif uname == ADMIN_USER:
                error = "Username already taken. Please choose another."
            elif uname in _load_users():
                error = f"Username '{uname}' is already registered."
            else:
                _save_user({
                    "username":      uname,
                    "password_hash": _hash(pwd),
                    "first_name":    first,
                    "last_name":     last,
                    "email":         email,
                    "role":          role,
                    "created_at":    _dt.datetime.now().isoformat(timespec="seconds"),
                })
                success = f"Account created for '{uname}'! You can now sign in."
                _log_user_interaction(
                    "register",
                    details={"role": role, "email": _clean_text(email, max_len=120)},
                    username=uname,
                )
        return render_template("register.html", error=error, success=success)

    @app.get("/logout")
    def logout():
        username = str(session.get("username", "") or "")
        if username:
            _log_user_interaction("logout", username=username)
        session.clear()
        return redirect(url_for("login"))

    @app.get("/health")
    def health():
        if state["artifact"] is None:
            return (
                jsonify({"status": "error", "message": state["error"], "model_path": str(model_path)}),
                500,
            )
        return jsonify({"status": "ok", "model_path": str(model_path)})

    @app.get("/doc")
    @app.get("/docs")
    def docs():
        return render_template("docs.html")

    @app.post("/predict")
    @api_login_required
    def predict():
        if state["artifact"] is None:
            _log_user_interaction("predict", "failed", details={"reason": "model_not_loaded"})
            return jsonify({"error": state["error"]}), 500

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            _log_user_interaction("predict", "failed", details={"reason": "invalid_json"})
            return jsonify({"error": "Invalid JSON body. Expected an object with feature values."}), 400

        feature_names = state["artifact"]["feature_names"]
        missing_features = [name for name in feature_names if name not in payload]
        if missing_features:
            _log_user_interaction(
                "predict",
                "failed",
                details={"reason": "missing_features", "missing_count": len(missing_features)},
            )
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
            _log_user_interaction("predict", "failed", details={"reason": "non_numeric_feature"})
            return jsonify({"error": f"All features must be numeric. {exc}"}), 400

        try:
            patient_details = _extract_patient_details(payload)
        except ValueError as exc:
            _log_user_interaction("predict", "failed", details={"reason": "invalid_patient_details"})
            return jsonify({"error": str(exc)}), 400

        feature_frame = pd.DataFrame([row], columns=feature_names)
        model = state["artifact"]["model"]

        prediction = int(model.predict(feature_frame)[0])
        probability = get_positive_probability(model, feature_frame)
        username = str(session.get("username", ""))

        response = {"prediction": prediction, "risk_probability": probability}
        if probability is None:
            response["note"] = "This model does not expose calibrated probability scores."
        try:
            _save_prediction(username, row, prediction, probability, patient_details=patient_details)
            summary = _build_prediction_summary(_load_predictions_for_user(username))
            response["prediction_count"] = summary["count"]
            response["comparison"] = {
                "delta_probability": summary["delta_probability"],
                "trend": summary["trend"],
            }
            response["patient_name"] = patient_details.get("patient_name", "")
        except Exception as exc:  # noqa: BLE001
            response["save_warning"] = f"Prediction generated, but history could not be saved: {exc}"
            _log_user_interaction("predict", "failed", details={"reason": "history_write_failed"})
        else:
            _log_user_interaction(
                "predict",
                details={
                    "prediction": prediction,
                    "risk_probability": probability,
                    "patient_name": _clean_text(patient_details.get("patient_name", ""), max_len=120),
                },
            )
        return jsonify(response)

    @app.post("/api/analyze-report")
    @api_login_required
    def analyze_report():
        if state["artifact"] is None:
            _log_user_interaction("analyze_report", "failed", details={"reason": "model_not_loaded"})
            return jsonify({"error": state["error"]}), 500

        try:
            uploaded = request.files.get("report_file")
            if uploaded is None:
                _log_user_interaction("analyze_report", "failed", details={"reason": "missing_report_file"})
                return jsonify({"error": "Missing report_file upload."}), 400

            report_text, report_name = _read_uploaded_report(uploaded)
            analysis = _analyze_report_text(report_text)
            patient_details = _extract_patient_details(request.form.to_dict())

            feature_names = state["artifact"]["feature_names"]
            row: dict[str, float] = {}
            missing_for_model: list[str] = []
            for feature_name in feature_names:
                raw = request.form.get(feature_name, "").strip()
                if raw == "":
                    missing_for_model.append(feature_name)
                    continue
                try:
                    row[feature_name] = float(raw)
                except ValueError:
                    _log_user_interaction("analyze_report", "failed", details={"reason": "non_numeric_feature"})
                    return jsonify({"error": f"Feature '{feature_name}' must be numeric."}), 400

            prediction = None
            probability = None
            prediction_count = None
            saved_to_history = False
            username = str(session.get("username", ""))

            if not missing_for_model:
                feature_frame = pd.DataFrame([row], columns=feature_names)
                model = state["artifact"]["model"]
                prediction = int(model.predict(feature_frame)[0])
                probability = get_positive_probability(model, feature_frame)
                _save_prediction(
                    username,
                    row,
                    prediction,
                    probability,
                    patient_details=patient_details,
                    report_context={
                        "report_file_name": report_name,
                        "report_analysis_level": analysis["analysis_level"],
                    },
                )
                prediction_count = _build_prediction_summary(_load_predictions_for_user(username))["count"]
                saved_to_history = True

            _log_user_interaction(
                "analyze_report",
                details={
                    "saved_to_history": saved_to_history,
                    "analysis_level": analysis.get("analysis_level"),
                    "missing_model_features": len(missing_for_model),
                },
            )

            return jsonify(
                {
                    "status": "success",
                    "file_name": report_name,
                    "analysis": analysis,
                    "prediction": prediction,
                    "risk_probability": probability,
                    "missing_model_features": missing_for_model,
                    "saved_to_history": saved_to_history,
                    "prediction_count": prediction_count,
                    "patient_name": patient_details.get("patient_name", ""),
                }
            )
        except ValueError as exc:
            _log_user_interaction("analyze_report", "failed", details={"reason": str(exc)[:120]})
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            _log_user_interaction("analyze_report", "failed", details={"reason": "unexpected_error"})
            return jsonify({"error": f"Unable to analyze report right now. {exc}"}), 500

    @app.get("/api/predictions")
    @api_login_required
    def get_predictions():
        username = str(session.get("username", ""))
        limit = request.args.get("limit", default=100, type=int)
        if limit is None or limit <= 0:
            limit = 100
        limit = min(limit, 500)

        predictions = _load_predictions_for_user(username)
        return jsonify(
            {
                "status": "success",
                "count": len(predictions),
                "data": predictions[:limit],
            }
        )

    @app.get("/api/predictions/summary")
    @api_login_required
    def get_prediction_summary():
        username = str(session.get("username", ""))
        predictions = _load_predictions_for_user(username)
        summary = _build_prediction_summary(predictions)
        return jsonify({"status": "success", **summary})

    @app.get("/api/predictions/export")
    @api_login_required
    def export_predictions():
        username = str(session.get("username", ""))
        predictions = _load_predictions_for_user(username)
        if not predictions:
            return jsonify({"error": "No predictions available to export."}), 404

        buffer = _io.StringIO()
        fieldnames = list(predictions[0].keys())
        writer = _csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
        csv_content = buffer.getvalue()
        buffer.close()

        response = Response(csv_content, mimetype="text/csv")
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{username}_prediction_history.csv"'
        )
        return response

    @app.get("/api/predictions/download")
    @api_login_required
    def download_prediction():
        username = str(session.get("username", ""))
        predictions = _load_predictions_for_user(username)
        if not predictions:
            return jsonify({"error": "No predictions found for download."}), 404

        entry = request.args.get("entry", default=0, type=int)
        if entry is None:
            entry = 0
        if entry < 0 or entry >= len(predictions):
            return jsonify({"error": "Requested prediction index is out of range."}), 400

        fmt = (request.args.get("format", default="txt", type=str) or "txt").lower()
        if fmt not in {"txt", "json", "csv"}:
            return jsonify({"error": "Unsupported format. Use txt, json, or csv."}), 400

        record = predictions[entry]
        stamp = str(record.get("submitted_at", "latest")).replace(":", "-")
        filename_base = f"{username}_prediction_{entry + 1}_{stamp}"

        if fmt == "json":
            json_text = _json.dumps(record, indent=2, ensure_ascii=False)
            response = Response(json_text, mimetype="application/json")
            response.headers["Content-Disposition"] = f'attachment; filename="{filename_base}.json"'
            return response

        if fmt == "csv":
            buffer = _io.StringIO()
            writer = _csv.DictWriter(buffer, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerow(record)
            response = Response(buffer.getvalue(), mimetype="text/csv")
            response.headers["Content-Disposition"] = f'attachment; filename="{filename_base}.csv"'
            return response

        report_text = _build_prediction_text_report(record)
        response = Response(report_text, mimetype="text/plain")
        response.headers["Content-Disposition"] = f'attachment; filename="{filename_base}.txt"'
        return response

    @app.get("/api/model-insights")
    def get_model_insights():
        if state["artifact"] is None:
            return jsonify({"error": state["error"]}), 500

        insights = _get_model_insights()
        return jsonify({"status": "success", **insights})

    @app.get("/api/patients")
    def get_patients():
        try:
            data_path = Path("data/diabetes.csv")
            if not data_path.exists():
                return jsonify({"error": "Patient data not found."}), 404

            df = pd.read_csv(data_path)

            status_filter = (request.args.get("status", default="all", type=str) or "all").lower()
            if status_filter in {"diabetic", "risk"}:
                df = df[df["Outcome"] == 1]
            elif status_filter in {"non-diabetic", "healthy", "non_diabetic"}:
                df = df[df["Outcome"] == 0]

            full_data = (request.args.get("full", default="0", type=str) or "0").lower() in {
                "1",
                "true",
                "yes",
            }
            total = int(len(df))

            if full_data:
                return jsonify(
                    {
                        "status": "success",
                        "data": df.to_dict(orient="records"),
                        "pagination": {
                            "page": 1,
                            "per_page": total,
                            "total": total,
                            "total_pages": 1,
                        },
                    }
                )

            page = request.args.get("page", default=1, type=int)
            per_page = request.args.get("per_page", default=10, type=int)
            if page is None or page < 1:
                page = 1
            if per_page is None or per_page < 1:
                per_page = 10
            per_page = min(per_page, 200)

            total_pages = max(1, int(_math.ceil(total / per_page))) if per_page else 1
            if page > total_pages:
                page = total_pages

            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            page_frame = df.iloc[start_idx:end_idx]

            return jsonify(
                {
                    "status": "success",
                    "data": page_frame.to_dict(orient="records"),
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total": total,
                        "total_pages": total_pages,
                    },
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/add-report")
    @api_login_required
    def add_report():
        """Accept a patient report JSON and persist it to SQLite."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            _log_user_interaction("add_report", "failed", details={"reason": "invalid_json"})
            return jsonify({"error": "Invalid JSON body."}), 400

        required = ["patient_name", "age", "Pregnancies", "Glucose",
                    "BloodPressure", "SkinThickness", "Insulin", "BMI",
                    "DiabetesPedigreeFunction", "notes"]
        missing = [k for k in required if k not in payload]
        if missing:
            _log_user_interaction(
                "add_report",
                "failed",
                details={"reason": "missing_fields", "missing_count": len(missing)},
            )
            return jsonify({"error": "Missing fields.", "missing": missing}), 400

        with _db_conn() as conn:
            conn.execute(
                """
                INSERT INTO reports (
                    submitted_by, submitted_at, patient_name, age, Pregnancies, Glucose,
                    BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(session.get("username", "")),
                    _dt.datetime.now().isoformat(timespec="seconds"),
                    _clean_text(payload.get("patient_name", ""), max_len=120),
                    _safe_int(payload.get("age")),
                    _to_sql_float(payload.get("Pregnancies")),
                    _to_sql_float(payload.get("Glucose")),
                    _to_sql_float(payload.get("BloodPressure")),
                    _to_sql_float(payload.get("SkinThickness")),
                    _to_sql_float(payload.get("Insulin")),
                    _to_sql_float(payload.get("BMI")),
                    _to_sql_float(payload.get("DiabetesPedigreeFunction")),
                    _clean_text(payload.get("notes", ""), max_len=400),
                ),
            )
            conn.commit()

        _log_user_interaction(
            "add_report",
            details={"patient_name": _clean_text(payload.get("patient_name", ""), max_len=120)},
        )
        return jsonify({"status": "success", "message": "Report saved successfully."})

    @app.get("/api/interactions")
    @api_login_required
    def get_interactions():
        username = str(session.get("username", "") or "")
        role = str(session.get("role", "patient") or "patient")
        event_type = _clean_text(request.args.get("event_type", ""), max_len=60)
        scope = (request.args.get("scope", default="mine", type=str) or "mine").lower()
        limit = request.args.get("limit", default=100, type=int)
        if limit is None or limit <= 0:
            limit = 100
        limit = min(limit, 500)

        clauses: list[str] = []
        params: list[object] = []
        if role != "admin" or scope != "all":
            clauses.append("username = ?")
            params.append(username)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)

        query = """
            SELECT
                id, username, event_type, status, request_method, request_path,
                ip_address, user_agent, details, created_at
            FROM user_interactions
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)

        with _db_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        data = []
        for row in rows:
            details_raw = row["details"] or ""
            details_payload: dict | str
            if not details_raw:
                details_payload = {}
            else:
                try:
                    details_payload = _json.loads(details_raw)
                except _json.JSONDecodeError:
                    details_payload = details_raw

            data.append(
                {
                    "id": row["id"],
                    "username": row["username"] or "",
                    "event_type": row["event_type"],
                    "status": row["status"],
                    "request_method": row["request_method"] or "",
                    "request_path": row["request_path"] or "",
                    "ip_address": row["ip_address"] or "",
                    "user_agent": row["user_agent"] or "",
                    "details": details_payload,
                    "created_at": row["created_at"],
                }
            )

        return jsonify({"status": "success", "count": len(data), "data": data})

    @app.get("/api/interactions/summary")
    @api_login_required
    def get_interactions_summary():
        username = str(session.get("username", "") or "")
        role = str(session.get("role", "patient") or "patient")
        scope = (request.args.get("scope", default="mine", type=str) or "mine").lower()

        clauses: list[str] = []
        params: list[object] = []
        if role != "admin" or scope != "all":
            clauses.append("username = ?")
            params.append(username)

        query = """
            SELECT event_type, status, COUNT(*) AS event_count
            FROM user_interactions
        """
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " GROUP BY event_type, status ORDER BY event_count DESC, event_type ASC"

        with _db_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        by_event: dict[str, dict[str, int]] = {}
        total = 0
        for row in rows:
            event = str(row["event_type"])
            event_status = str(row["status"])
            count = int(row["event_count"])
            total += count
            if event not in by_event:
                by_event[event] = {}
            by_event[event][event_status] = count

        return jsonify({"status": "success", "total_events": total, "by_event": by_event})

    @app.get("/api/reports")
    @api_login_required
    def get_reports():
        """Return saved patient reports for the logged-in user."""
        username = str(session.get("username", ""))
        try:
            with _db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        submitted_by, submitted_at, patient_name, age, Pregnancies, Glucose,
                        BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, notes
                    FROM reports
                    WHERE submitted_by = ?
                    ORDER BY submitted_at DESC, id DESC
                    """,
                    (username,),
                ).fetchall()

            data = []
            for row in rows:
                data.append(
                    {
                        "submitted_by": row["submitted_by"],
                        "submitted_at": row["submitted_at"],
                        "patient_name": row["patient_name"],
                        "age": row["age"],
                        "Pregnancies": row["Pregnancies"],
                        "Glucose": row["Glucose"],
                        "BloodPressure": row["BloodPressure"],
                        "SkinThickness": row["SkinThickness"],
                        "Insulin": row["Insulin"],
                        "BMI": row["BMI"],
                        "DiabetesPedigreeFunction": row["DiabetesPedigreeFunction"],
                        "notes": row["notes"],
                    }
                )
            return jsonify({"status": "success", "data": data})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
