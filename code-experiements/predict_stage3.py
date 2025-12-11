#!/usr/bin/env python3
"""
Stage-3: Parallel, chunked prediction on a new dataset with
classification + regression aware outputs.

This script:

- Shares TASK_MODE with Stage-1/Stage-2:
    * "classification": all y_* treated as classification targets.
    * "regression": all y_* treated as regression targets.

- Uses models trained in Stage-2:
    Classification models (files in trained_models/):
        y_<target>_RF.joblib
        y_<target>_LGBM.joblib
        y_<target>_XGB.joblib
        y_<target>_HGB.joblib
        y_<target>_CB.joblib
        y_<target>_best.joblib

    Regression models:
        y_<target>_RF_REG.joblib
        y_<target>_LGBM_REG.joblib
        y_<target>_XGB_REG.joblib
        y_<target>_HGB_REG.joblib
        y_<target>_CB_REG.joblib
        y_<target>_best.joblib

- Uses model_cv_results_parallel.csv from Stage-2 to pull metrics:
    Classification:
        RF_f1_macro, RF_precision_macro, RF_recall_macro, RF_accuracy, RF_auc, ...
    Regression:
        RF_REG_rmse, RF_REG_mae, RF_REG_r2, ...

- Modes:
    PREDICTION_MODE = "best"
        Classification:
            - For each target y_<name>:
                * loads y_<name>_best.joblib
                * uses Stage-2 row to find best_model and metrics.
            - Outputs per row:
                * y_<name>                     (actual, if present)
                * y_<name>_interpolated_model4 (predicted class)
                * y_<name>_interpolated_model4_metric1 (max class prob)
                * y_<name>_interpolated_model4_metric2_f1
                * y_<name>_interpolated_model4_metric3_recall
                * y_<name>_interpolated_model4_metric4_precision
                * y_<name>_interpolated_model4_metric5_auc
              (metrics repeated per row from Stage-2 CV).

        Regression:
            - For each target y_<name>:
                * loads y_<name>_best.joblib
                * uses Stage-2 row to find best_model and metrics.
            - Outputs per row:
                * y_<name>                             (actual, if present)
                * y_<name>_interpolated_model4         (predicted value)
                * y_<name>_interpolated_model4_residual (y_actual - y_pred, if actual exists)
                * y_<name>_interpolated_model4_metric1_rmse
                * y_<name>_interpolated_model4_metric2_mae
                * y_<name>_interpolated_model4_metric3_r2
              (metrics repeated per row from Stage-2 CV).

    PREDICTION_MODE = "all_models"
        Classification:
            - For each target y_<name>, attempts to load:
                y_<name>_RF.joblib
                y_<name>_LGBM.joblib
                y_<name>_XGB.joblib
                y_<name>_HGB.joblib
                y_<name>_CB.joblib
            - For each available model M:
                * y_<name>_M_interpolated_model4
                * y_<name>_M_interpolated_model4_metric1
                * y_<name>_M_interpolated_model4_metric2_f1
                * y_<name>_M_interpolated_model4_metric3_recall
                * y_<name>_M_interpolated_model4_metric4_precision
                * y_<name>_M_interpolated_model4_metric5_auc
              (metrics pulled from Stage-2 row for that model, if present).

        Regression:
            - For each target y_<name>, attempts to load any of:
                y_<name>_RF_REG.joblib, y_<name>_LGBM_REG.joblib, ...
            - For each available regression model M_REG:
                * y_<name>_M_REG_interpolated_model4
                * y_<name>_M_REG_interpolated_model4_residual
                * y_<name>_M_REG_interpolated_model4_metric1_rmse
                * y_<name>_M_REG_interpolated_model4_metric2_mae
                * y_<name>_M_REG_interpolated_model4_metric3_r2
              (metrics pulled from Stage-2 row for that model, if present).

Inputs:
    - new_data.csv               (full feature table, may or may not include y_* columns)
    - feature_importances/feature_importances_<y>.csv
    - trained_models/*.joblib
    - trained_models/model_cv_results_parallel.csv

Output:
    - stage3_predictions.csv (id_pwd_id + predictions and metrics for all targets)

Logging:
    - logs/y_<target>_stage3.log one file per target.

Note:
    - For CatBoostEncoder in Stage-3, if y_<target> exists with labels,
      we fit the encoder on labeled rows and transform all rows.
      If no labels exist for a target, we drop object columns for numeric models.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from joblib import Parallel, delayed, load

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

# Optional MLflow (often off for Stage-3)
try:
    import mlflow  # type: ignore[import]

    mlflow_available = True
except Exception:  # pragma: no cover - optional
    mlflow = None  # type: ignore[assignment]
    mlflow_available = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Global task mode (must match Stage-1 & Stage-2)
#   "classification" or "regression"
TASK_MODE = "classification"  # or "regression"

# Prediction mode:
#   "best"       → only use y_<target>_best.joblib
#   "all_models" → use all available model artifacts per target
PREDICTION_MODE = "best"  # or "all_models"

data_path = "new_data.csv"
feature_importance_dir = Path("feature_importances")
models_dir = Path("trained_models")
log_dir = Path("logs")

id_prefix = "id_"
feature_prefix = "ft_"
target_prefix = "y_"

random_state = 42

# Chunking for large new datasets
chunk_size = 50_000

cpu_count = os.cpu_count() or 4
n_jobs_targets = max(min(cpu_count - 1, 16), 2)

use_catboost_encoder = True
cat_fill_value = "NA_CAT"

# MLflow toggle for Stage-3 (often False)
use_mlflow = False
mlflow_experiment_name = "stage3_scoring"

predictions_output_path = Path("stage3_predictions.csv")

feature_importance_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

cv_results_csv = models_dir / "model_cv_results_parallel.csv"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage3")


def get_target_logger(y_col: str) -> logging.Logger:
    """Create a per-target logger writing to logs/y_<target>_stage3.log."""
    logger = logging.getLogger(f"stage3.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "_stage3_file", False)
        for handler in logger.handlers
    )
    if not exists:
        file_handler = logging.FileHandler(
            log_dir / f"{y_col}_stage3.log",
            mode="w",
            encoding="utf-8",
        )
        file_handler._stage3_file = True  # type: ignore[attr-defined]
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"),
        )
        logger.addHandler(file_handler)

    logger.propagate = True
    return logger


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Detect id_, ft_, y_ columns by prefix."""
    id_cols = [c for c in df.columns if c.startswith(id_prefix)]
    ft_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    y_cols = [c for c in df.columns if c.startswith(target_prefix)]
    return id_cols, ft_cols, y_cols


def load_cv_results() -> Optional[pd.DataFrame]:
    """Load Stage-2 CV results if available."""
    if not cv_results_csv.exists():
        log.warning("Stage-2 CV results not found at %s", cv_results_csv)
        return None
    return pd.read_csv(cv_results_csv)


def get_cv_row_for_target(
    df_cv: Optional[pd.DataFrame],
    y_col: str,
) -> Optional[pd.Series]:
    """Return the CV results row for a given target."""
    if df_cv is None:
        return None
    subset = df_cv[df_cv["target"] == y_col]
    if subset.empty:
        return None
    return subset.iloc[0]


def load_feature_importances_for_target(
    y_col: str,
    task_mode: str,
    logger: logging.Logger,
) -> Optional[List[str]]:
    """Load Stage-1 importance file for y_col and select usable features."""
    path = feature_importance_dir / f"feature_importances_{y_col}.csv"
    if not path.exists():
        logger.warning("No feature_importances file for %s at %s", y_col, path)
        return None

    df_imp = pd.read_csv(path)
    if "feature_name" not in df_imp.columns:
        logger.warning(
            "feature_importances_%s.csv has no 'feature_name' column; skipping.",
            y_col,
        )
        return None

    if task_mode == "regression":
        if "RF_REG" not in df_imp.columns:
            logger.warning(
                "feature_importances_%s.csv has no 'RF_REG'; skipping.",
                y_col,
            )
            return None
        mask = df_imp["RF_REG"] > 0
    else:
        required = ["RF", "LGBM", "CB", "XGB", "HGB"]
        missing = [c for c in required if c not in df_imp.columns]
        if missing:
            logger.warning(
                "feature_importances_%s.csv missing columns %s; skipping.",
                y_col,
                missing,
            )
            return None
        mask = (
            (df_imp["RF"] > 0)
            & (df_imp["LGBM"] > 0)
            & (df_imp["CB"] > 0)
            & (df_imp["XGB"] > 0)
            & (df_imp["HGB"] > 0)
        )

    selected = df_imp.loc[mask, "feature_name"].dropna().unique().tolist()
    if not selected:
        logger.warning(
            "No features selected for %s after importance filter; skipping.",
            y_col,
        )
        return None

    logger.info(
        "Stage-3: selected %d features for %s from Stage-1 importances.",
        len(selected),
        y_col,
    )
    return selected


def prepare_views_classification_for_prediction(
    X: pd.DataFrame,
    y_optional: Optional[pd.Series],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare numeric + CatBoost views for classification predictions.

    If y_optional is provided and has labels, we fit CatBoostEncoder on
    labeled rows and transform all rows. Otherwise, fallback: drop object
    columns for numeric view.
    """
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if use_catboost_encoder and cat_cols and y_optional is not None:
        mask = y_optional.notna()
        if mask.any():
            logger.info(
                "Fitting CatBoostEncoder on %d labeled rows for prediction.",
                int(mask.sum()),
            )
            encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
            encoder = encoder.fit(X_num.loc[mask, :], y_optional.loc[mask])
            X_num = encoder.transform(X_num)
        else:
            logger.info(
                "No labeled rows for this target in Stage-3; dropping object "
                "columns for numeric view.",
            )
            X_num = X_num.drop(columns=cat_cols)
    elif cat_cols:
        logger.info(
            "No labels or CatBoostEncoder disabled; dropping object columns.",
        )
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    X_cb = X.copy()
    cb_cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        X_cb[col] = X_cb[col].astype("string").fillna(cat_fill_value)

    return X_num, X_cb


def prepare_view_regression_for_prediction(
    X: pd.DataFrame,
    y_optional: Optional[pd.Series],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Prepare numeric CatBoost-encoded view for regression predictions."""
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if use_catboost_encoder and cat_cols and y_optional is not None:
        mask = y_optional.notna()
        if mask.any():
            logger.info(
                "Fitting CatBoostEncoder on %d labeled rows for regression.",
                int(mask.sum()),
            )
            encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
            encoder = encoder.fit(X_num.loc[mask, :], y_optional.loc[mask])
            X_num = encoder.transform(X_num)
        else:
            logger.info(
                "No labels available; dropping object columns for regression X.",
            )
            X_num = X_num.drop(columns=cat_cols)
    elif cat_cols:
        logger.info(
            "No labels or CatBoostEncoder disabled; dropping object columns.",
        )
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    return X_num


# ---------------------------------------------------------------------------
# MODEL LOADING HELPERS
# ---------------------------------------------------------------------------

def available_classification_models(y_col: str) -> Dict[str, Path]:
    """Return available classification model paths for a target."""
    names = ["RF", "LGBM", "XGB", "HGB", "CB"]
    paths: Dict[str, Path] = {}
    for name in names:
        path = models_dir / f"{y_col}_{name}.joblib"
        if path.exists():
            paths[name] = path
    return paths


def available_regression_models(y_col: str) -> Dict[str, Path]:
    """Return available regression model paths for a target."""
    names = ["RF_REG", "LGBM_REG", "XGB_REG", "HGB_REG", "CB_REG"]
    paths: Dict[str, Path] = {}
    for name in names:
        path = models_dir / f"{y_col}_{name}.joblib"
        if path.exists():
            paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# PER-TARGET PREDICTION
# ---------------------------------------------------------------------------

def predict_for_target(
    y_col: str,
    df: pd.DataFrame,
    id_cols: List[str],
    base_ft_cols: List[str],
    df_cv: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Run Stage-3 scoring for a single target."""
    logger = get_target_logger(y_col)
    logger.info(
        "=== Stage-3 scoring for %s (task_mode=%s, prediction_mode=%s) ===",
        y_col,
        TASK_MODE,
        PREDICTION_MODE,
    )

    # Determine selected features from Stage-1
    selected = load_feature_importances_for_target(
        y_col=y_col,
        task_mode=TASK_MODE,
        logger=logger,
    )
    if not selected:
        return None

    # Keep features that exist in new_data
    use_features = [c for c in base_ft_cols if c in selected and c in df.columns]
    if not use_features:
        logger.warning(
            "No usable features for %s in new_data; skipping.",
            y_col,
        )
        return None

    X_full = df[use_features].copy()
    y_actual: Optional[pd.Series]
    if y_col in df.columns:
        y_actual = df[y_col]
    else:
        y_actual = None

    cv_row = get_cv_row_for_target(df_cv, y_col)

    # Determine model files
    best_model_path = models_dir / f"{y_col}_best.joblib"
    if not best_model_path.exists():
        logger.warning("Best model for %s not found at %s", y_col, best_model_path)

    if TASK_MODE == "regression":
        model_paths = available_regression_models(y_col)
    else:
        model_paths = available_classification_models(y_col)

    if PREDICTION_MODE == "best":
        if not best_model_path.exists():
            logger.warning(
                "Prediction mode 'best' but best model missing for %s; skipping.",
                y_col,
            )
            return None
        model_paths = {}  # we only use best
    else:
        if not model_paths:
            logger.warning(
                "Prediction mode 'all_models' but no individual models for %s; "
                "trying best model only.",
                y_col,
            )
            if not best_model_path.exists():
                return None

    # Prepare feature views
    if TASK_MODE == "classification":
        X_num, X_cb = prepare_views_classification_for_prediction(
            X_full,
            y_optional=y_actual,
            logger=logger,
        )
    else:
        X_num = prepare_view_regression_for_prediction(
            X_full,
            y_optional=y_actual,
            logger=logger,
        )
        X_cb = X_full  # not used in regression paths

    n_rows = df.shape[0]
    chunks = list(range(0, n_rows, chunk_size))

    out_frames: List[pd.DataFrame] = []

    # Base output frame with IDs (once)
    base_out = pd.DataFrame(index=df.index)
    for col in id_cols:
        if col in df.columns:
            base_out[col] = df[col]

    if y_actual is not None:
        base_out[y_col] = y_actual

    # Classification branch
    if TASK_MODE == "classification":
        if PREDICTION_MODE == "best":
            if cv_row is None:
                logger.warning(
                    "No CV row for %s; metrics columns will be NaN.",
                    y_col,
                )
                best_name = None
            else:
                best_name = cv_row.get("best_model")
                if isinstance(best_name, float) and np.isnan(best_name):
                    best_name = None

            model = load(best_model_path)
            all_preds: List[np.ndarray] = []
            all_probs: List[np.ndarray] = []

            for start in chunks:
                end = min(start + chunk_size, n_rows)
                logger.info("Scoring %s rows [%d, %d)", y_col, start, end)
                idx = X_num.index[start:end]
                if isinstance(model, CatBoostClassifier):
                    preds = model.predict(X_cb.loc[idx, :])
                    proba = model.predict_proba(X_cb.loc[idx, :])
                elif isinstance(model, HistGradientBoostingClassifier):
                    proba = model.predict_proba(X_num.loc[idx, :])
                    preds = np.argmax(proba, axis=1)
                elif isinstance(model, LGBMClassifier):
                    proba = model.predict_proba(X_num.loc[idx, :])
                    preds = np.argmax(proba, axis=1)
                elif isinstance(model, RandomForestClassifier):
                    proba = model.predict_proba(X_num.loc[idx, :])
                    preds = model.predict(X_num.loc[idx, :])
                elif isinstance(model, XGBClassifier):
                    proba = model.predict_proba(X_num.loc[idx, :])
                    preds = np.argmax(proba, axis=1)
                else:
                    raise ValueError(f"Unsupported best model type for {y_col}: {type(model)}")

                all_preds.append(preds)
                all_probs.append(proba)

            y_pred = np.concatenate(all_preds)
            proba_all = np.concatenate(all_probs, axis=0)
            max_proba = proba_all.max(axis=1)

            out = base_out.copy()
            col_pred = f"{y_col}_interpolated_model4"
            col_prob = f"{y_col}_interpolated_model4_metric1"
            out[col_pred] = y_pred
            out[col_prob] = max_proba

            # Attach CV metrics (if available)
            if cv_row is not None and best_name is not None:
                prefix = f"{best_name}_"
                f1_val = cv_row.get(f"{prefix}f1_macro", np.nan)
                rec_val = cv_row.get(f"{prefix}recall_macro", np.nan)
                prec_val = cv_row.get(f"{prefix}precision_macro", np.nan)
                acc_val = cv_row.get(f"{prefix}accuracy", np.nan)
                auc_val = cv_row.get(f"{prefix}auc", np.nan)

                out[f"{y_col}_interpolated_model4_metric2_f1"] = f1_val
                out[f"{y_col}_interpolated_model4_metric3_recall"] = rec_val
                out[f"{y_col}_interpolated_model4_metric4_precision"] = prec_val
                out[f"{y_col}_interpolated_model4_metric5_auc"] = auc_val
            else:
                out[f"{y_col}_interpolated_model4_metric2_f1"] = np.nan
                out[f"{y_col}_interpolated_model4_metric3_recall"] = np.nan
                out[f"{y_col}_interpolated_model4_metric4_precision"] = np.nan
                out[f"{y_col}_interpolated_model4_metric5_auc"] = np.nan

            out_frames.append(out)

        else:
            # all_models
            models_available = available_classification_models(y_col)
            if not models_available and best_model_path.exists():
                models_available = {"BEST": best_model_path}
            if not models_available:
                logger.warning("No models available for %s; skipping.", y_col)
                return None

            out = base_out.copy()

            for name, path in models_available.items():
                logger.info("Scoring %s with model %s", y_col, name)
                model = load(path)
                all_preds: List[np.ndarray] = []
                all_probs: List[np.ndarray] = []

                for start in chunks:
                    end = min(start + chunk_size, n_rows)
                    idx = X_num.index[start:end]
                    if isinstance(model, CatBoostClassifier):
                        preds = model.predict(X_cb.loc[idx, :])
                        proba = model.predict_proba(X_cb.loc[idx, :])
                    elif isinstance(model, HistGradientBoostingClassifier):
                        proba = model.predict_proba(X_num.loc[idx, :])
                        preds = np.argmax(proba, axis=1)
                    elif isinstance(model, LGBMClassifier):
                        proba = model.predict_proba(X_num.loc[idx, :])
                        preds = np.argmax(proba, axis=1)
                    elif isinstance(model, RandomForestClassifier):
                        proba = model.predict_proba(X_num.loc[idx, :])
                        preds = model.predict(X_num.loc[idx, :])
                    elif isinstance(model, XGBClassifier):
                        proba = model.predict_proba(X_num.loc[idx, :])
                        preds = np.argmax(proba, axis=1)
                    else:
                        raise ValueError(
                            f"Unsupported classification model type for {y_col}: {type(model)}",
                        )

                    all_preds.append(preds)
                    all_probs.append(proba)

                y_pred = np.concatenate(all_preds)
                proba_all = np.concatenate(all_probs, axis=0)
                max_proba = proba_all.max(axis=1)

                prefix_cols = f"{y_col}_{name}_interpolated_model4"
                out[prefix_cols] = y_pred
                out[f"{prefix_cols}_metric1"] = max_proba

                # attach metrics from CV if available
                if cv_row is not None and name in ["RF", "LGBM", "XGB", "HGB", "CB"]:
                    metric_prefix = f"{name}_"
                    f1_val = cv_row.get(f"{metric_prefix}f1_macro", np.nan)
                    rec_val = cv_row.get(f"{metric_prefix}recall_macro", np.nan)
                    prec_val = cv_row.get(f"{metric_prefix}precision_macro", np.nan)
                    acc_val = cv_row.get(f"{metric_prefix}accuracy", np.nan)
                    auc_val = cv_row.get(f"{metric_prefix}auc", np.nan)
                else:
                    f1_val = rec_val = prec_val = acc_val = auc_val = np.nan

                out[f"{prefix_cols}_metric2_f1"] = f1_val
                out[f"{prefix_cols}_metric3_recall"] = rec_val
                out[f"{prefix_cols}_metric4_precision"] = prec_val
                out[f"{prefix_cols}_metric5_auc"] = auc_val

            out_frames.append(out)

        logger.info("Finished Stage-3 scoring for %s", y_col)
        return out_frames[0]

    # Regression branch
    if PREDICTION_MODE == "best":
        if not best_model_path.exists():
            logger.warning(
                "Prediction mode 'best' but best model missing for %s; skipping.",
                y_col,
            )
            return None

        model = load(best_model_path)
        all_preds_reg: List[np.ndarray] = []

        for start in chunks:
            end = min(start + chunk_size, n_rows)
            idx = X_num.index[start:end]
            if isinstance(model, (CatBoostRegressor, LGBMRegressor, XGBRegressor)):
                preds = model.predict(X_num.loc[idx, :])
            elif isinstance(model, (RandomForestRegressor, HistGradientBoostingRegressor)):
                preds = model.predict(X_num.loc[idx, :])
            else:
                raise ValueError(f"Unsupported regression model type for {y_col}: {type(model)}")

            all_preds_reg.append(preds)

        y_pred = np.concatenate(all_preds_reg)
        out = base_out.copy()
        col_pred = f"{y_col}_interpolated_model4"
        out[col_pred] = y_pred

        if y_actual is not None:
            out[f"{y_col}_interpolated_model4_residual"] = y_actual - y_pred
        else:
            out[f"{y_col}_interpolated_model4_residual"] = np.nan

        if cv_row is not None:
            best_name = cv_row.get("best_model")
            if isinstance(best_name, float) and np.isnan(best_name):
                best_name = None
            if best_name is not None:
                prefix = f"{best_name}_"
                rmse_val = cv_row.get(f"{prefix}rmse", np.nan)
                mae_val = cv_row.get(f"{prefix}mae", np.nan)
                r2_val = cv_row.get(f"{prefix}r2", np.nan)
            else:
                rmse_val = mae_val = r2_val = np.nan
        else:
            rmse_val = mae_val = r2_val = np.nan

        out[f"{y_col}_interpolated_model4_metric1_rmse"] = rmse_val
        out[f"{y_col}_interpolated_model4_metric2_mae"] = mae_val
        out[f"{y_col}_interpolated_model4_metric3_r2"] = r2_val

        logger.info("Finished Stage-3 regression scoring for %s", y_col)
        return out

    # Regression, all_models
    models_available = available_regression_models(y_col)
    if not models_available and best_model_path.exists():
        models_available = {"BEST_REG": best_model_path}
    if not models_available:
        logger.warning("No regression models available for %s; skipping.", y_col)
        return None

    out = base_out.copy()

    for name, path in models_available.items():
        logger.info("Scoring %s with regression model %s", y_col, name)
        model = load(path)
        all_preds_reg = []

        for start in chunks:
            end = min(start + chunk_size, n_rows)
            idx = X_num.index[start:end]
            if isinstance(model, (CatBoostRegressor, LGBMRegressor, XGBRegressor)):
                preds = model.predict(X_num.loc[idx, :])
            elif isinstance(model, (RandomForestRegressor, HistGradientBoostingRegressor)):
                preds = model.predict(X_num.loc[idx, :])
            else:
                raise ValueError(
                    f"Unsupported regression model type for {y_col}: {type(model)}",
                )

            all_preds_reg.append(preds)

        y_pred = np.concatenate(all_preds_reg)
        prefix_cols = f"{y_col}_{name}_interpolated_model4"
        out[prefix_cols] = y_pred

        if y_actual is not None:
            out[f"{prefix_cols}_residual"] = y_actual - y_pred
        else:
            out[f"{prefix_cols}_residual"] = np.nan

        if cv_row is not None and name in [
            "RF_REG",
            "LGBM_REG",
            "XGB_REG",
            "HGB_REG",
            "CB_REG",
        ]:
            metric_prefix = f"{name}_"
            rmse_val = cv_row.get(f"{metric_prefix}rmse", np.nan)
            mae_val = cv_row.get(f"{metric_prefix}mae", np.nan)
            r2_val = cv_row.get(f"{metric_prefix}r2", np.nan)
        else:
            rmse_val = mae_val = r2_val = np.nan

        out[f"{prefix_cols}_metric1_rmse"] = rmse_val
        out[f"{prefix_cols}_metric2_mae"] = mae_val
        out[f"{prefix_cols}_metric3_r2"] = r2_val

    logger.info("Finished Stage-3 regression scoring for %s", y_col)
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Stage-3 predictions over all y_* targets on new_data.csv."""
    log.info("Loading new data from %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)

    id_cols, ft_cols, y_cols = detect_columns(df)
    log.info(
        "Detected %d id_, %d ft_, %d y_ columns in new_data.",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    df_cv = load_cv_results()

    if use_mlflow and mlflow_available and mlflow is not None:
        mlflow.set_experiment(mlflow_experiment_name)

    # Determine targets from models_dir (y_*_best.joblib) to ensure we only
    # score targets that were trained in Stage-2.
    model_best_files = list(models_dir.glob("y_*_best.joblib"))
    targets_from_models = sorted(
        {p.name.split("_best.joblib")[0] for p in model_best_files},
    )

    if not targets_from_models:
        log.error("No y_*_best.joblib models found in %s; aborting Stage-3.", models_dir)
        return

    log.info(
        "Stage-3 will score %d targets with n_jobs_targets=%d (task_mode=%s, "
        "prediction_mode=%s).",
        len(targets_from_models),
        n_jobs_targets,
        TASK_MODE,
        PREDICTION_MODE,
    )

    results = Parallel(n_jobs=n_jobs_targets)(
        delayed(predict_for_target)(
            y_col,
            df,
            id_cols,
            ft_cols,
            df_cv,
        )
        for y_col in targets_from_models
    )

    frames = [r for r in results if r is not None]
    if not frames:
        log.error("No predictions generated in Stage-3.")
        return

    # Combine side-by-side on index
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(frame.drop(columns=id_cols, errors="ignore"), how="outer")

    combined.to_csv(predictions_output_path, index=False)
    log.info("Stage-3 predictions saved to %s", predictions_output_path)


if __name__ == "__main__":
    main()
