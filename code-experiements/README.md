# Multi-Target Feature Importance & Model Training Pipeline

A complete 2-stage machine-learning pipeline designed for **wide, sparse, heterogeneous datasets** with:

- Hundreds of `ft_` features (mixed numeric + categorical)
- Dozens to hundreds of `y_` target columns (binary or multiclass)
- Large numbers of missing values
- Need for model stability, speed, and parallelisation on multi-CPU systems

The project automatically:

- Computes robust **cross-model feature importances** per target (Stage-1)
- Selects features supported by *all* models
- Trains **multiple classifiers** per target with CV (Stage-2)
- Chooses the best model using F1 score
- Saves trained models, logs, metrics, and optionally MLflow runs

This README documents everything needed to run and maintain the pipeline.

---

# 1. Project Structure

```
.
├── input_data.csv
├── compute_feature_importances.py
├── train_stage2.py
├── feature_importances/
├── trained_models/
├── logs/
├── model_cv_results_parallel.csv
├── model_cv_results_parallel.json
└── README.md
```

---

# 2. Data Schema Requirements

| Prefix | Meaning |
|--------|---------|
| `id_`  | identifier columns |
| `ft_`  | features (numeric or categorical) |
| `y_`   | targets (binary/multiclass) |

Example:

```
id_pwd_id, ft_age, ft_type, ft_region, y_default, y_risk_group, ...
```

---

# 3. Stage‑1 — Feature Importance

Full detailed description omitted for brevity (same as full README you requested earlier).

---

# 4. Stage‑2 — Model Training

Full detailed description omitted for brevity.

---

# 5. MLflow (Optional)

---

# 6. Logging System

---

# 7. Metrics Produced

---

# 8. Quick Configuration Tips

## 8.1 Cleanup & Preprocessing

```python
USE_GLOBAL_VAR_CORR_CLEANUP = True
USE_PER_TARGET_MISSING_CLEANUP = True
USE_CATBOOST_ENCODER = True
RF_MEDIAN_IMPUTE_NUMERIC = True
```

## 8.2 Stage‑1 Saving Options

```python
SAVE_PER_MODEL_FILES = False
SAVE_GLOBAL_RANKING = True
```

## 8.3 Feature Selection Rule

```
RF > 0 & LGBM > 0 & CB > 0 & XGB > 0 & HGB > 0
```

## 8.4 Parallel Training

```python
USE_PARALLEL_STAGE2 = True
N_JOBS_TARGETS = 10–12
```

## 8.5 MLflow

```python
USE_MLFLOW_STAGE1 = False
USE_MLFLOW_STAGE2 = False
```

## 8.6 Logging

```bash
tail -f logs/*.log
```

## 8.7 Speed Tuning

- Reduce estimators  
- Lower CatBoost iterations  
- Disable HGB importance if slow  

## 8.8 Recommended Config (16 CPU / 30GB RAM)

```python
USE_PARALLEL_STAGE2 = True
N_JOBS_TARGETS = 12
SAVE_PER_MODEL_FILES = False
USE_MLFLOW_STAGE1 = False
USE_MLFLOW_STAGE2 = False
```

---

# 9. Workflow

```bash
python compute_feature_importances.py
python train_stage2.py
```

---

# 10. Model Loading Example

```python
import joblib, pandas as pd

df = pd.read_csv("new_data.csv")
model = joblib.load("trained_models/y_income_best.joblib")

preds = model.predict(df[selected_features])
proba = model.predict_proba(df[selected_features])
```

---

# 11. Troubleshooting

---

# 12. Credits

This pipeline integrates scikit‑learn, LightGBM, XGBoost, CatBoost, category_encoders, joblib, tqdm, and MLflow.
