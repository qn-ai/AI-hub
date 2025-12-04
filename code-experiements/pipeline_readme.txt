1. Folder layout

Put everything in a project folder, e.g.:

ml_training_pipeline/
│
├── config.py
├── logger.py
├── mlflow_utils.py
├── feature_selection.py
├── preprocessing.py
├── cv_evaluation.py
├── train_target.py
├── train_all.py
│
├── input_data.csv                 # your full dataset
├── feature_importances/           # all your FI CSVs
│     ├── feature_importances_y_xxx.csv         (combined file, used)
│     ├── feature_importances_y_xxx_CB.csv      (ignored)
│     ├── feature_importances_y_xxx_XGB.csv     (ignored)
│     ├── feature_importances_y_xxx_RF.csv      (ignored)
│     ├── feature_importances_y_xxx_LGBM.csv    (ignored)
│     └── ...
├── trained_models/                # created after training
├── logs/                          # created after training
└── mlruns/                        # MLflow tracking data (created)

input_data.csv must contain:
	•	id_ columns → identifiers
	•	ft_ columns → features (numeric + categorical)
	•	y_ columns → targets (156+ targets in your case)

The exact column names don’t matter as long as they follow those prefixes.

⸻

2. Install dependencies

From inside ml_training_pipeline/:

pip install pandas numpy scikit-learn lightgbm xgboost catboost category-encoders mlflow joblib

(On Posit Workbench: do this in your Python environment.)

⸻

3. Check config.py

Key lines to confirm:

DATA_PATH = "input_data.csv"        # full dataset
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
MODELS_DIR = Path("trained_models")

MIN_SAMPLES_PER_TARGET = 200        # you can reduce if needed
N_SPLITS = 3                        # CV folds

Parallelism is auto-tuned:

_CPU_COUNT = os.cpu_count() or 4
AUTO_JOBS = max(min(_CPU_COUNT // 2, 8), 2)
N_JOBS_TARGETS = AUTO_JOBS

On a 16-CPU box this will usually use 8 parallel targets.

⸻

4. How training works (conceptually)

For each y_ column:
	1.	Select features from feature_importances/:
	•	Look at all CSV files whose names contain that y_col
	•	Ignore files ending in _CB.csv, _XGB.csv, _RF.csv, _LGBM.csv
	•	Pick the largest remaining file (the “combined” importance file)
	•	Read it (blank first column is the feature name index)
	•	Keep only rows where RF > 0 AND LGBM > 0 AND CB > 0 AND XGB > 0
	•	Keep only those features that exist in your ft_ columns
	2.	Filter the data:
	•	Use only rows where that y_col is not null
	•	Use only the selected feature columns
	3.	Prepare target y:
	•	If y is string → LabelEncoder to turn it into integers
	•	If y is numeric → use as is
	4.	Split features into numeric vs categorical:
	•	Categorical = dtype object/string
	•	Numeric = everything else
	5.	Train 4 models with CV:
	•	RF / LGBM / XGBoost use the numeric pipeline:
	•	SimpleImputer(median) on numeric
	•	CatBoostEncoder on categorical
	•	Model sees clean numeric matrix
	•	CatBoost uses raw categoricals:
	•	Categorical columns cast to string, NaN → "NA_CAT"
	•	Numeric columns left as numeric with NaNs
	•	CatBoost gets cat_features indices and handles NaNs + categories itself
	6.	Metrics per model:
	•	F1_macro
	•	precision_macro
	•	recall_macro
	•	accuracy
	•	roc_auc_macro_ovr (if possible)
	•	log_loss
	•	train_time_total_sec
	•	plus: n_features_used, n_samples_used, n_cv_folds, n_param_configs
	7.	Pick best model by highest F1_macro.
	8.	Train best model on full data for that target.
	9.	Save package to trained_models/<y_col>_best.joblib, including:
	•	pipeline_type: "numeric" or "catboost_raw"
	•	model / pipeline
	•	label_encoder (or None)
	•	features list
	•	cat_cols (for catboost_raw)
	•	best_params, meta
	10.	Log everything to MLflow:
	•	Run per target (train_y_xxx)
	•	Params, metrics, best_f1_macro
	•	Feature list as artifact
	•	Best numeric pipeline as MLflow model (for numeric case)

All targets are processed in parallel with joblib.

⸻

5. How to run training

From inside ml_training_pipeline/:

python train_all.py

Watch progress:
	•	Console output (and logs go to logs/pipeline.log)
	•	Errors per target won’t crash the whole run; they just log and skip.

⸻

6. Outputs to look at

6.1 CV results: model_cv_results_parallel.csv

Each row is “one model on one target”, e.g.:

target	model	f1_macro	precision_macro	recall_macro	accuracy	roc_auc_macro_ovr	log_loss	train_time_total_sec	n_features_used	n_samples_used	n_cv_folds	n_param_configs
y_age	RF	0.72	…	…	…	…	…	…	45	3200	3	1
y_age	LGBM	0.75	…	…	…	…	…	…	45	3200	3	1
y_age	XGB	0.74	…	…	…	…	…	…	45	3200	3	1
y_age	CAT	0.78	…	…	…	…	…	…	45	3200	3	1
y_income	RF	…	…	…	…	…	…	…	60	2800	3	1
…	…	…	…	…	…	…	…	…	…	…	…	…

Use this file to:
	•	compare models (RF vs LGBM vs XGB vs CAT) for each y_
	•	check how many features & samples were used
	•	sanity-check training time and AUC

Same content is in JSON: model_cv_results_parallel.json.

⸻

6.2 Resource summary: training_resource_summary.json

Example:

{
  "total_wall_time_sec": 1234.56,
  "targets_trained": 120,
  "cpu_count": 16,
  "n_jobs_targets": 8
}

Gives you a high-level sense of how heavy the run was.

⸻

6.3 Trained models: trained_models/*.joblib

One file per target:
	•	trained_models/y_age_best.joblib
	•	trained_models/y_income_best.joblib
	•	…

Each is a dict like:

{
  "pipeline_type": "numeric" or "catboost_raw",
  "pipeline": <sklearn Pipeline>      # for numeric
  "model": <CatBoostClassifier>       # for catboost_raw
  "label_encoder": <LabelEncoder or None>,
  "features": ["ft_x", "ft_y", ...],
  "cat_cols": [...],                  # only for catboost_raw
  "target": "y_age",
  "model_name": "RF" / "LGBM" / "XGB" / "CAT",
  "best_params": {...},
}


⸻

6.4 MLflow tracking

MLflow stores everything in ./mlruns/.

To inspect:

mlflow ui --backend-store-uri file:./mlruns

Then open the URL it prints (usually http://127.0.0.1:5000) in your browser.

You’ll see:
	•	runs like train_y_age, train_y_income, etc.
	•	metrics (F1, AUC, recall, etc.)
	•	parameters (n_estimators, depth, etc.)
	•	artifacts (feature list, sometimes model)

Very handy for comparing targets & models visually.
