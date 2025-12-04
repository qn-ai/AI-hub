# mlflow_utils.py
from pathlib import Path
from typing import Any, Dict, List

import mlflow

from config import MODELS_DIR


def init_mlflow(experiment_name: str = "multi_target_training") -> None:
    """Initialise MLflow tracking in local folder ./mlruns."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)


def start_target_run(y_col: str):
    """Start a new MLflow run for a given target."""
    return mlflow.start_run(run_name=f"train_{y_col}")


def log_params(params: Dict[str, Any], prefix: str | None = None) -> None:
    """Log parameters to MLflow, optionally with a prefix."""
    if prefix:
        params = {f"{prefix}_{k}": v for k, v in params.items()}
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], prefix: str | None = None) -> None:
    """Log scalar metrics to MLflow, optionally with a prefix."""
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            # ignore non-float values
            pass


def log_feature_list(y_col: str, features: List[str]) -> None:
    """Save feature list for a target and log as MLflow artifact."""
    MODELS_DIR.mkdir(exist_ok=True)
    feature_file = MODELS_DIR / f"{y_col}_features.txt"
    with open(feature_file, "w", encoding="utf-8") as f:
        for ft in features:
            f.write(f"{ft}\n")
    mlflow.log_artifact(str(feature_file))


def log_sklearn_model(model, name: str) -> None:
    """Log a sklearn-like model (pipeline or estimator) to MLflow."""
    mlflow.sklearn.log_model(model, artifact_path=name)
