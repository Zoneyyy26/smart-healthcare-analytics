# Smart Healthcare Analytics System

## Overview
This project trains a baseline machine-learning model to predict diabetes risk from patient metrics and serves predictions through a Flask API.

## What is Included
- Data preprocessing utilities (`src/utils/preprocess.py`)
- Deterministic synthetic dataset generator for demo/testing (`src/utils/data_factory.py`)
- Training pipeline with model selection (Logistic Regression vs Random Forest) (`src/train.py`)
- Flask API for health check and prediction (`src/app.py`)
- Automated tests for preprocessing, training, and API behavior (`tests/`)

## Repository Structure
- `src/` - training and API code
- `src/utils/` - preprocessing and dataset generation helpers
- `docs/` - data summary artifacts
- `notebooks/` - exploratory notebook(s)
- `tests/` - automated tests
- `data/` - local datasets (ignored in git)
- `models/` - trained artifacts (ignored in git)

## Quick Start (Windows PowerShell)
1. Install dependencies:
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Validate environment:
```powershell
.\venv\Scripts\python.exe setup_environment.py
```

3. Train model:
```powershell
.\venv\Scripts\python.exe -m src.train
```

If `data/diabetes.csv` is missing, training auto-generates a synthetic dataset at that path.

4. Run API:
```powershell
.\venv\Scripts\python.exe -m src.app
```

## API Endpoints
- `GET /health`
- `POST /predict`
- `POST /api/analyze-report`
- `GET /api/predictions`
- `GET /api/predictions/summary`
- `GET /api/interactions`
- `GET /api/interactions/summary`

## Database
- Default SQLite database path: `data/user_interactions.db`
- Core tables: `users`, `predictions`, `reports`, `user_interactions`

Example request body for `/predict`:

```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 90,
  "BMI": 31.2,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 35
}
```

## Train with a Real Dataset
If you have a real dataset, place it at `data/diabetes.csv` (target column: `Outcome`) and run:

```powershell
.\venv\Scripts\python.exe -m src.train --dataset-path data/diabetes.csv --target-col Outcome
```

## Run Tests
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Notes
- Synthetic data is for demo/testing only and is not clinical evidence.
- The Flask server is development-grade and should be replaced with a production WSGI server for deployment.
