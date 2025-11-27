# config.py
import os
import mlflow

RANDOM_STATE = 42
N_JOBS = -1
TEST_SIZE = 0.2
N_OUTER_FOLDS = 5  # CV folds after tuning
EXPERIMENT_NAME = "multitarget_rf_lgbm_catboost"

# Path for local results
RESULTS_CSV_PATH = "multitarget_model_comparison_results.csv"
AGG_PLOT_PATH = "multitarget_model_comparison_plot.png"

# -----------------------------
# MLflow configuration
# -----------------------------
def setup_mlflow():
    """
    Configure MLflow tracking.
    - Tracking URI (local file-based by default)
    - You can point this to a remote server whose artifact root is S3.
    """

    # Example A: local tracking (models saved locally via MLflow)
    mlflow.set_tracking_uri("file:./mlruns")

    # Example B (commented): remote MLflow server with S3 artifact root
    # mlflow.set_tracking_uri("http://your-mlflow-server:5000")
    # Then start server with:
    #   mlflow server \
    #     --backend-store-uri sqlite:///mlflow.db \
    #     --default-artifact-root s3://your-bucket/path

    mlflow.set_experiment(EXPERIMENT_NAME)


# -----------------------------
# Model spaces (hyperparameters)
# -----------------------------
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models_and_spaces():
    models_and_spaces = {
        "RandomForest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            "param_distributions": {
                "n_estimators": [200, 400, 600],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5],
            },
        },
        "LightGBM": {
            "estimator": LGBMClassifier(
                objective="multiclass",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            "param_distributions": {
                "n_estimators": [300, 600, 900],
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 10, 20],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
            },
        },
        "CatBoost": {
            "estimator": CatBoostClassifier(
                loss_function="MultiClass",
                verbose=False,
                random_seed=RANDOM_STATE,
            ),
            "param_distributions": {
                "depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5, 7],
                "iterations": [300, 600, 900],
            },
        },
    }
    return models_and_spaces
