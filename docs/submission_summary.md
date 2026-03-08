# Smart Healthcare Analytics - Submission Summary

## Project Objective
This project delivers a baseline healthcare analytics pipeline that predicts diabetes risk from patient features and exposes predictions through an API. The goal is to support early risk screening using a reproducible data-science workflow.

## Final Scope Delivered
- A reproducible ML training pipeline in `src/train.py`
- Data preprocessing helpers in `src/utils/preprocess.py`
- Synthetic dataset generator for demo and fallback workflow in `src/utils/data_factory.py`
- Flask inference API in `src/app.py`
- Environment verification script in `setup_environment.py`
- Automated tests in `tests/`
- Updated run and usage documentation in `README.md`

## Technical Approach
### 1) Data Handling and Preprocessing
- Input expected as CSV with target column `Outcome`
- Features are all non-target columns
- Known invalid zeros are treated as missing for:
  - `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
- Median imputation and standardization are applied in the model pipeline

### 2) Model Training
- Candidate models:
  - Logistic Regression
  - Random Forest Classifier
- Model selection uses 5-fold stratified cross-validation with ROC-AUC
- Best model is trained on train split and evaluated on test split
- Saved artifact includes:
  - trained pipeline
  - ordered feature names
  - selected model name
  - CV ROC-AUC and test metrics

### 3) API Layer
- `GET /health`: model availability and server health
- `POST /predict`: single-sample prediction from JSON payload
- Input validation added:
  - required feature checks
  - numeric conversion checks
- Probability handling supports both `predict_proba` and `decision_function` fallback

## Reliability and Developer Experience Improvements
- `setup_environment.py` now provides stable, ASCII-safe output and returns a non-zero code when dependencies are missing
- Training now supports automatic synthetic dataset generation when `data/diabetes.csv` is absent, making the project runnable out of the box
- Test suite added for preprocessing, training artifact generation, and API behavior

## Validation Results
Validated in local venv with:
- `./venv/Scripts/python.exe setup_environment.py`
- `./venv/Scripts/python.exe -m src.train --model-output models/diabetes_model.joblib`
- `./venv/Scripts/python.exe -m unittest discover -s tests -v`

Observed training output (latest run):
- Selected model: Logistic Regression
- Cross-val ROC-AUC: 0.8435
- Test Accuracy: 0.8117
- Test ROC-AUC: 0.8852

## Known Constraints
- Synthetic dataset is for demonstration/testing only and not clinical evidence
- Flask server is development-grade; production deployment should use a proper WSGI server

## Submission Readiness
As of March 8, 2026, the project is submission-ready for an academic/demo context with:
- reproducible setup,
- functioning training and inference paths,
- automated tests,
- and updated documentation.
