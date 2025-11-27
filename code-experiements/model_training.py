# model_training.py
from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from config import N_JOBS, N_OUTER_FOLDS, RANDOM_STATE
from data_preprocessing import (
    build_sklearn_preprocessor,
    detect_feature_types,
    get_catboost_cat_indices,
    train_val_split_for_target,
)


def evaluate_holdout(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    target_name: str,
) -> tuple[Dict[str, float], np.ndarray, str]:
    """
    Compute metrics on a hold-out validation set.
    Metric keys are prefixed with val_{target}_{model}_.
    """
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        has_proba = True
    else:
        y_proba = None
        has_proba = False

    prefix = f"val_{target_name}_{model_name}_"
    metrics: Dict[str, float] = {}

    metrics[f"{prefix}accuracy"] = accuracy_score(y_val, y_pred)
    metrics[f"{prefix}f1_macro"] = f1_score(
        y_val,
        y_pred,
        average="macro",
    )
    metrics[f"{prefix}f1_weighted"] = f1_score(
        y_val,
        y_pred,
        average="weighted",
    )
    metrics[f"{prefix}precision_macro"] = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics[f"{prefix}precision_weighted"] = precision_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    metrics[f"{prefix}recall_macro"] = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics[f"{prefix}recall_weighted"] = recall_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    if has_proba:
        try:
            metrics[f"{prefix}roc_auc_ovr"] = roc_auc_score(
                y_val,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            metrics[f"{prefix}roc_auc_ovr"] = np.nan

        try:
            metrics[f"{prefix}log_loss"] = log_loss(y_val, y_proba)
        except Exception:
            metrics[f"{prefix}log_loss"] = np.nan
    else:
        metrics[f"{prefix}roc_auc_ovr"] = np.nan
        metrics[f"{prefix}log_loss"] = np.nan

    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    return metrics, cm, report


def evaluate_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    target_name: str,
    n_folds: int = N_OUTER_FOLDS,
) -> Dict[str, float]:
    """
    Run k-fold cross-validation and return mean/std metrics.
    Metric keys are prefixed with cv_{target}_{model}_.
    """
    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "precision_weighted": "precision_weighted",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=N_JOBS,
        return_train_score=False,
    )

    metrics_cv: Dict[str, float] = {}
    prefix = f"cv_{target_name}_{model_name}_"

    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")
        metrics_cv[f"{prefix}{metric_name}_mean"] = float(np.mean(values))
        metrics_cv[f"{prefix}{metric_name}_std"] = float(np.std(values))

    return metrics_cv


def tune_and_train_one_model_for_target(
    model_name: str,
    base_estimator,
    param_distributions: dict,
    df: pd.DataFrame,
    X_full: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    For one target + one model:
      - train/val split (dropping missing labels)
      - hyperparameter tuning via RandomizedSearchCV
      - hold-out & CV evaluation
      - MLflow logging
      - return a flat dict of metrics/params for CSV.
    """
    X_train_raw, X_val_raw, y_train, y_val, label_encoder = train_val_split_for_target(
        df,
        X_full,
        target_col,
    )

    preprocessor_sklearn, _, cat_cols = build_sklearn_preprocessor(X_full)
    cat_indices = get_catboost_cat_indices(X_full, cat_cols)

    if model_name in ("RandomForest", "LightGBM"):
        estimator = base_estimator
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor_sklearn),
                ("model", estimator),
            ],
        )
        search_estimator = pipe
        param_dist_with_prefix = {
            f"model__{k}": v for k, v in param_distributions.items()
        }
        fit_params = {}
    else:
        search_estimator = base_estimator
        param_dist_with_prefix = param_distributions
        fit_params = {"cat_features": cat_indices}

    search = RandomizedSearchCV(
        estimator=search_estimator,
        param_distributions=param_dist_with_prefix,
        n_iter=15,
        scoring="f1_macro",
        n_jobs=N_JOBS,
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    search.fit(X_train_raw, y_train, **fit_params)
    best_model = search.best_estimator_
    best_params = search.best_params_

    val_metrics, cm, report = evaluate_holdout(
        best_model,
        X_val_raw,
        y_val,
        model_name,
        target_col,
    )

    mask_all = df[target_col].notna()
    X_all_target = X_full.loc[mask_all]
    y_all_target = df[target_col].loc[mask_all]

    if label_encoder is not None and y_all_target.dtype == "object":
        y_all_target = label_encoder.transform(y_all_target)

    cv_metrics = evaluate_cv(
        best_model,
        X_all_target,
        y_all_target,
        model_name,
        target_col,
    )

    run_name = f"{target_col}_{model_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target_col", target_col)
        mlflow.log_params(best_params)

        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(cv_metrics)

        cm_df = pd.DataFrame(cm)
        cm_path = f"cm_{target_col}_{model_name}.csv"
        cm_df.to_csv(cm_path, index=False)
        mlflow.log_artifact(cm_path)

        report_path = f"classification_report_{target_col}_{model_name}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        if model_name == "RandomForest":
            mlflow.sklearn.log_model(best_model, artifact_path="model")
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(best_model, artifact_path="model")
        elif model_name == "CatBoost":
            mlflow.catboost.log_model(best_model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

    row: dict = {
        "target": target_col,
        "model": model_name,
        "best_params": json.dumps(best_params),
    }
    row.update(val_metrics)
    row.update(cv_metrics)
    return row
