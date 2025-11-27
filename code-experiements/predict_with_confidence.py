# predict_with_confidence.py
"""
Use the tuned results from run_multitarget_experiment.py to:
- pick the best model per target (RandomForest / LightGBM / CatBoost),
- retrain it on all labeled rows for that target,
- compute per-row confidence intervals for the predicted class probability:
    * pmax_mean (mean predicted prob for chosen class)
    * pmax_ci_low / pmax_ci_high (e.g. 95% CI),
- and save a CSV with id + targets + predictions + CI metrics.

Assumes:
  - config.py
  - data_preprocessing.py
  - run_multitarget_experiment.py
already exist.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import RESULTS_CSV_PATH, RANDOM_STATE, get_models_and_spaces
from data_preprocessing import (
    build_sklearn_preprocessor,
    detect_feature_types,
    get_catboost_cat_indices,
    load_data,
    split_features_targets,
)


def pick_best_model_for_target(
    results_df: pd.DataFrame,
    target_col: str,
) -> Dict[str, object]:
    """
    Select the best model for a given target based on
    cv_{target}_{model}_f1_macro_mean.
    """
    sub = results_df[results_df["target"] == target_col]
    if sub.empty:
        raise ValueError(f"No results found in CSV for target '{target_col}'")

    best_row: pd.Series | None = None
    best_score = -np.inf

    for _, row in sub.iterrows():
        model_name = row["model"]
        metric_col = f"cv_{target_col}_{model_name}_f1_macro_mean"
        if metric_col not in sub.columns:
            continue

        score = row[metric_col]
        if pd.isna(score):
            continue

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        raise ValueError(
            f"No valid cv F1 macro scores found for target '{target_col}'",
        )

    model_name = best_row["model"]
    best_params = json.loads(best_row["best_params"])

    return {
        "model_name": model_name,
        "best_params": best_params,
        "cv_f1_macro": best_score,
    }


def train_rf_pipeline_for_target(
    best_params: Dict[str, object],
    X_full: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """
    Rebuild and fit a RandomForest + preprocessor pipeline on all labeled rows
    for this target, using tuned hyperparameters.
    """
    models_spaces = get_models_and_spaces()
    base_estimator = models_spaces["RandomForest"]["estimator"]

    rf_params = {
        k.split("model__", 1)[1]: v
        for k, v in best_params.items()
        if k.startswith("model__")
    }
    base_estimator.set_params(**rf_params)

    mask = y.notna()
    X_train = X_full.loc[mask].copy()
    y_train = y.loc[mask].copy()

    preprocessor, _, _ = build_sklearn_preprocessor(X_full)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_estimator),
        ],
    )
    pipe.fit(X_train, y_train)
    return pipe


def rf_ci_from_pipeline(
    rf_pipeline: Pipeline,
    X_full: pd.DataFrame,
    ci_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-row confidence intervals for the predicted class probability
    using per-tree bootstrap from a fitted RF pipeline.
    """
    from sklearn.ensemble import RandomForestClassifier

    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf: RandomForestClassifier = rf_pipeline.named_steps["model"]

    X_trans = preprocessor.transform(X_full)

    all_tree_probs = np.stack(
        [tree.predict_proba(X_trans) for tree in rf.estimators_],
        axis=0,
    )  # (n_trees, n_samples, n_classes)

    mean_probs = all_tree_probs.mean(axis=0)  # (n_samples, n_classes)

    alpha = 1.0 - ci_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    lower = np.percentile(all_tree_probs, lower_q, axis=0)
    upper = np.percentile(all_tree_probs, upper_q, axis=0)

    pred_class = mean_probs.argmax(axis=1)
    rows = np.arange(mean_probs.shape[0])

    pmax_mean = mean_probs[rows, pred_class]
    ci_low = lower[rows, pred_class]
    ci_high = upper[rows, pred_class]

    return pred_class, pmax_mean, ci_low, ci_high


def train_lgbm_ensemble_for_target(
    best_params: Dict[str, object],
    X_full: pd.DataFrame,
    y: pd.Series,
    n_models: int = 5,
) -> List[Pipeline]:
    """
    Train an ensemble of LightGBM pipelines (same tuned hyperparameters,
    different seeds) on all labeled rows for this target.
    """
    from lightgbm import LGBMClassifier

    models_spaces = get_models_and_spaces()
    base_estimator = models_spaces["LightGBM"]["estimator"]

    lgbm_params = {
        k.split("model__", 1)[1]: v
        for k, v in best_params.items()
        if k.startswith("model__")
    }

    mask = y.notna()
    X_train = X_full.loc[mask].copy()
    y_train = y.loc[mask].copy()

    preprocessor, _, _ = build_sklearn_preprocessor(X_full)

    ensemble: List[Pipeline] = []

    for i in range(n_models):
        params = lgbm_params.copy()
        params["random_state"] = RANDOM_STATE + i
        model = LGBMClassifier(**params)

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ],
        )
        pipe.fit(X_train, y_train)
        ensemble.append(pipe)

    return ensemble


def train_catboost_ensemble_for_target(
    best_params: Dict[str, object],
    X_full: pd.DataFrame,
    y: pd.Series,
    n_models: int = 5,
):
    """
    Train an ensemble of CatBoost models (same tuned hyperparameters,
    different seeds) on all labeled rows for this target.
    """
    from catboost import CatBoostClassifier

    mask = y.notna()
    X_train = X_full.loc[mask].copy()
    y_train = y.loc[mask].copy()

    _, cat_cols = detect_feature_types(X_full)
    cat_indices = get_catboost_cat_indices(X_full, cat_cols)

    ensemble = []

    for i in range(n_models):
        params = best_params.copy()
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("verbose", False)

        model = CatBoostClassifier(
            random_seed=RANDOM_STATE + i,
            **params,
        )
        model.fit(X_train, y_train, cat_features=cat_indices)
        ensemble.append(model)

    return ensemble


def ensemble_ci_from_models(
    models,
    X_full: pd.DataFrame,
    ci_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-row confidence intervals from an ensemble of models that support
    predict_proba(X_full).
    """
    all_probs = np.stack(
        [m.predict_proba(X_full) for m in models],
        axis=0,
    )  # (n_models, n_samples, n_classes)

    mean_probs = all_probs.mean(axis=0)

    alpha = 1.0 - ci_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    lower = np.percentile(all_probs, lower_q, axis=0)
    upper = np.percentile(all_probs, upper_q, axis=0)

    pred_class = mean_probs.argmax(axis=1)
    rows = np.arange(mean_probs.shape[0])

    pmax_mean = mean_probs[rows, pred_class]
    ci_low = lower[rows, pred_class]
    ci_high = upper[rows, pred_class]

    return pred_class, pmax_mean, ci_low, ci_high


def main() -> None:
    """Run prediction + CI generation for all multi-target columns."""
    df = load_data()
    X_full, target_cols, id_cols = split_features_targets(df, target_prefix="y_")

    print("Targets:", target_cols)
    print("ID columns:", id_cols)

    results_df = pd.read_csv(RESULTS_CSV_PATH)

    out_cols: List[str] = []
    if id_cols:
        out_cols.extend(id_cols)
    out_cols.extend(target_cols)
    df_out = df[out_cols].copy()

    for target_col in target_cols:
        print(f"\n=== Processing predictions with CIs for target: {target_col} ===")

        n_non_missing = df[target_col].notna().sum()
        if n_non_missing < 50:
            print(f"  Skipping {target_col}: only {n_non_missing} labeled rows.")
            continue

        best_info = pick_best_model_for_target(results_df, target_col)
        model_name = best_info["model_name"]
        best_params = best_info["best_params"]

        print(
            f"  Best model for {target_col}: {model_name} "
            f"(cv_f1_macro={best_info['cv_f1_macro']:.4f})",
        )

        y_target = df[target_col]

        if model_name == "RandomForest":
            rf_pipeline = train_rf_pipeline_for_target(
                best_params,
                X_full,
                y_target,
            )
            pred_class, pmax_mean, ci_low, ci_high = rf_ci_from_pipeline(
                rf_pipeline,
                X_full,
                ci_level=0.95,
            )
        elif model_name == "LightGBM":
            lgbm_ensemble = train_lgbm_ensemble_for_target(
                best_params,
                X_full,
                y_target,
                n_models=5,
            )
            pred_class, pmax_mean, ci_low, ci_high = ensemble_ci_from_models(
                lgbm_ensemble,
                X_full,
                ci_level=0.95,
            )
        elif model_name == "CatBoost":
            cat_ensemble = train_catboost_ensemble_for_target(
                best_params,
                X_full,
                y_target,
                n_models=5,
            )
            pred_class, pmax_mean, ci_low, ci_high = ensemble_ci_from_models(
                cat_ensemble,
                X_full,
                ci_level=0.95,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        base = f"{target_col}_{model_name}"
        df_out[f"{base}_pred"] = pred_class
        df_out[f"{base}_pmax_mean"] = pmax_mean
        df_out[f"{base}_pmax_ci_low"] = ci_low
        df_out[f"{base}_pmax_ci_high"] = ci_high

    out_path = "multitarget_predictions_with_confidence_intervals.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved predictions + confidence intervals to {out_path}")


if __name__ == "__main__":
    main()
