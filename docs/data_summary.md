# Data Summary

## Dataset Snapshot
The summary below corresponds to a diabetes-style dataset with 768 rows and 8 input features plus binary target `Outcome`.

| Metric | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| count | 768.0 | 768.0 | 768.0 | 768.0 | 768.0 | 768.0 | 768.0 | 768.0 | 768.0 |
| mean | 3.845 | 120.895 | 69.105 | 20.536 | 79.799 | 31.993 | 0.472 | 33.241 | 0.349 |
| std | 3.370 | 31.973 | 19.356 | 15.952 | 115.244 | 7.884 | 0.331 | 11.760 | 0.477 |
| min | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.078 | 21.0 | 0.0 |
| 25% | 1.0 | 99.0 | 62.0 | 0.0 | 0.0 | 27.3 | 0.244 | 24.0 | 0.0 |
| 50% | 3.0 | 117.0 | 72.0 | 23.0 | 30.5 | 32.0 | 0.372 | 29.0 | 0.0 |
| 75% | 6.0 | 140.25 | 80.0 | 32.0 | 127.25 | 36.6 | 0.626 | 41.0 | 1.0 |
| max | 17.0 | 199.0 | 122.0 | 99.0 | 846.0 | 67.1 | 2.42 | 81.0 | 1.0 |

## Preprocessing Logic Used
- Target: `Outcome`
- Features are all non-target columns
- Zero values are treated as missing for these columns:
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
- Missing values are imputed with median values inside the training pipeline
- Features are standardized before model fitting
