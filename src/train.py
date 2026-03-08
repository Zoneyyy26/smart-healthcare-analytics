from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.data_factory import generate_synthetic_diabetes_dataset
from src.utils.preprocess import load_dataset, normalize_feature_frame, split_features_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a chronic disease risk model.")
    parser.add_argument(
        "--dataset-path",
        default="data/diabetes.csv",
        help="Path to CSV dataset. Default: data/diabetes.csv",
    )
    parser.add_argument(
        "--target-col",
        default="Outcome",
        help="Target column in dataset. Default: Outcome",
    )
    parser.add_argument(
        "--model-output",
        default="models/diabetes_model.joblib",
        help="Output path for trained model artifact.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=768,
        help="Rows to generate when dataset is missing and auto-generation is enabled.",
    )
    parser.add_argument(
        "--auto-generate-if-missing",
        action="store_true",
        default=True,
        help="Generate synthetic dataset if --dataset-path does not exist. Default: enabled.",
    )
    parser.add_argument(
        "--no-auto-generate-if-missing",
        action="store_false",
        dest="auto_generate_if_missing",
        help="Disable synthetic data auto-generation when dataset is missing.",
    )
    return parser


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, x_test, y_test) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }


def select_best_model(x_train, y_train, random_state: int) -> tuple[str, Pipeline, float]:
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced"
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for name, model in candidates.items():
        pipeline = make_pipeline(model)
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")
        mean_score = float(scores.mean())
        if mean_score > best_score:
            best_name = name
            best_score = mean_score
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No candidate model was selected.")
    return best_name, best_pipeline, best_score


def main() -> None:
    args = build_parser().parse_args()

    try:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            if args.auto_generate_if_missing:
                generated = generate_synthetic_diabetes_dataset(
                    output_path=dataset_path,
                    n_samples=args.synthetic_samples,
                    random_state=args.random_state,
                )
                print(
                    f"Dataset not found. Generated synthetic dataset at: {dataset_path} "
                    f"({len(generated)} rows)"
                )
            else:
                raise FileNotFoundError(
                    f"Dataset not found at '{dataset_path}'. "
                    "Use --auto-generate-if-missing or provide a real dataset."
                )

        df = load_dataset(dataset_path)
        x, y = split_features_target(df, args.target_col)
        x = normalize_feature_frame(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        model_name, best_pipeline, cv_score = select_best_model(
            x_train, y_train, random_state=args.random_state
        )
        best_pipeline.fit(x_train, y_train)

        metrics = evaluate_model(best_pipeline, x_test, y_test)
        artifact = {
            "model": best_pipeline,
            "feature_names": list(x.columns),
            "target_col": args.target_col,
            "metrics": metrics,
            "selected_model": model_name,
            "cv_roc_auc": cv_score,
        }

        output_path = Path(args.model_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output_path)

        print(f"Selected model: {model_name}")
        print(f"Cross-val ROC-AUC: {cv_score:.4f}")
        print("Test metrics:")
        print(json.dumps(metrics, indent=2))
        print(f"Saved artifact to: {output_path}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
