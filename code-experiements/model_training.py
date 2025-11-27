# model_training.py
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost

from config import RANDOM_STATE, N_JOBS, N_OUTER_FOLDS
from data_preprocessing import (
    build_sklearn_preprocessor,
    detect_feature_types,
    get_catboost_cat_indices,
)


# -----------------------------
# HELPER: evaluation on hold-out
# -----------------------------
def evaluate_holdout(model, X_val, y_val, model_name, target_name):
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        has_proba = True
    else:
        y_proba = None
        has_proba = False

    metrics = {}
    prefix = f"val_{target_name}_{model_name}_"

    metrics[prefix + "accuracy"] = accuracy_score(y_val, y_pred)
    metrics[prefix + "f1_macro"] = f1_score(y_val, y_pred, average="macro")
    metrics[prefix + "f1_weighted"] = f1_score(y_val, y_pred, average="weighted")
    metrics[prefix + "precision_macro"] = precision_score(
        y_val, y_pred, average="macro", zero_division=0
    )
    metrics[prefix + "precision_weighted"] = precision_score(
        y_val, y_pred, average="weighted", zero_division=0
    )
    metrics[prefix + "recall_macro"] = recall_score(
        y_val, y_pred, average="macro", zero_division=0
    )
    metrics[prefix + "recall_weighted"] = recall_score(
        y_val, y_pred, average="weighted", zero_division=0
    )

    if has_proba:
        try:
            metrics[prefix + "roc_auc_ovr"] = roc_auc_score(
                y_val, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            metrics[prefix + "roc_auc_ovr"] = np.nan

        try:
            metrics[prefix + "log_loss"] = log_loss(y_val, y_proba)
        except Exception:
            metrics[prefix + "log_loss"] = np.nan
    else:
        metrics[prefix + "roc_auc_ovr"] = np.nan
        metrics[prefix + "log_loss"] = np.nan

    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    return metrics, cm, report


# -----------------------------
# HELPER: cross-validation
# -----------------------------
def evaluate_cv(model, X, y, model_name, target_name, n_folds=N_OUTER_FOLDS):
    cv = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
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

    metrics_cv = {}
    prefix = f"cv_{target_name}_{model_name}_"
    for key, values in cv_results.items():
        if key.startswith("test_"):
            metric_name = key.replace("test_", "")
            metrics_cv[prefix + metric_name + "_mean"] = np.mean(values)
            metrics_cv[prefix + metric_name + "_std"] = np.std(values)

    return metrics_cv


# -----------------------------
# MAIN: tune and train for one target & model
# -----------------------------
def tune_and_train_one_model_for_target(
    model_name,
    base_estimator,
    param_distributions,
    df,
    X_full,
    target_col,
):
    """
    For one target + one model:
      - build appropriate preprocessing
      - tune hyperparameters (RandomizedSearchCV)
      - evaluate hold-out and CV
      - log to MLflow
      - return metrics row for CSV
    """

    from data_preprocessing import train_val_split_for_target

    # Split train/val for this target (drop missing labels)
    X_train_raw, X_val_raw, y_train, y_val, label_encoder = train_val_split_for_target(
        df, X_full, target_col
    )

    # Build preprocessors
    preprocessor_sklearn, numeric_cols, cat_cols = build_sklearn_preprocessor(X_full)
    cat_indices = get_catboost_cat_indices(X_full, cat_cols)

    # For RF & LGBM: use sklearn preprocessor -> pipeline
    if model_name in ("RandomForest", "LightGBM"):
        from sklearn.pipeline import Pipeline

        estimator = base_estimator
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor_sklearn),
                ("model", estimator),
            ]
        )
        search_estimator = pipe
        param_dist_with_prefix = {
            "model__" + k: v for k, v in param_distributions.items()
        }
    else:
        # CatBoost: use raw X, handle categorical indices directly.
        # We'll still impute missing numeric values if you want,
        # but CatBoost can handle missing values in many cases.
        search_estimator = base_estimator
        param_dist_with_prefix = param_distributions

    from config import RANDOM_STATE

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

    # Fit search
    if model_name in ("RandomForest", "LightGBM"):
        search.fit(X_train_raw, y_train)
        best_model = search.best_estimator_
    else:
        # CatBoost needs cat_features indices
        cat_feat_idx = cat_indices
        search.fit(X_train_raw, y_train, model__cat_features=cat_feat_idx) if False else \
            search.fit(X_train_raw, y_train, cat_features=cat_feat_idx)
        best_model = search.best_estimator_

    best_params = search.best_params_

    # Hold-out metrics
    val_metrics, cm, report = evaluate_holdout(
        best_model, X_val_raw, y_val, model_name, target_col
    )

    # CV metrics on all non-missing rows for this target
    mask = df[target_col].notna()
    X_all_target = X_full.loc[mask]
    y_all_target = df[target_col].loc[mask]
    if label_encoder is not None and y_all_target.dtype == "object":
        y_all_target = label_encoder.transform(y_all_target)

    cv_metrics = evaluate_cv(
        best_model, X_all_target, y_all_target, model_name, target_col
    )

    # -------------------------
    # MLflow logging
    # -------------------------
    run_name = f"{target_col}_{model_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target_col", target_col)
        mlflow.log_params(best_params)

        # metrics
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(cv_metrics)

        # confusion matrix & classification report
        cm_df = pd.DataFrame(cm)
        cm_path = f"cm_{target_col}_{model_name}.csv"
        cm_df.to_csv(cm_path, index=False)
        mlflow.log_artifact(cm_path)

        report_path = f"classification_report_{target_col}_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log model
        if model_name == "RandomForest":
            mlflow.sklearn.log_model(best_model, artifact_path="model")
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(best_model, artifact_path="model")
        elif model_name == "CatBoost":
            mlflow.catboost.log_model(best_model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

    # Build a flat metrics row
    row = {
        "target": target_col,
        "model": model_name,
        "best_params": json.dumps(best_params),
    }
    row.update(val_metrics)
    row.update(cv_metrics)
    return row
