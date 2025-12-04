# config.py
from pathlib import Path
import os

# -----------------------
# Paths
# -----------------------
DATA_PATH = "input_data_top150_features.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
MODELS_DIR = Path("trained_models")

RESULTS_CSV_PATH = "model_cv_results_parallel.csv"
RESULTS_JSON_PATH = "model_cv_results_parallel.json"
RESOURCE_SUMMARY_PATH = "training_resource_summary.json"

# -----------------------
# Column prefixes
# -----------------------
ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

# -----------------------
# Training settings
# -----------------------
MIN_SAMPLES_PER_TARGET = 200
N_SPLITS = 3
RANDOM_STATE = 42

# -----------------------
# Parallel settings
# -----------------------
_CPU_COUNT = os.cpu_count() or 4
AUTO_JOBS = max(min(_CPU_COUNT // 2, 8), 2)  # good for up to 16 CPUs
N_JOBS_TARGETS = AUTO_JOBS

# -----------------------
# Param grids (1 config/model to save compute)
# -----------------------
RF_PARAM_GRID = [
    {"n_estimators": 300, "max_depth": None, "max_features": "sqrt"},
]

LGBM_PARAM_GRID = [
    {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.04},
]

XGB_PARAM_GRID = [
    {
        "n_estimators": 400,
        "max_depth": 7,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
]

CAT_PARAM_GRID = [
    {"iterations": 400, "depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 4.0}
]
