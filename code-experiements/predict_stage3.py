#!/usr/bin/env python3
"""
Stage-3: Parallel, chunked prediction on a new dataset (classification only).

Modes
-----

PREDICTION_MODE = "best"
    - For each target y_<name>, loads y_<name>_best.joblib.
    - Outputs:
        y_<name>                              (actual, if present)
        y_<name>_interpolated_model4          (predicted label)
        y_<name>_interpolated_model4_metric1  (row-level probability)
        y_<name>_interpolated_model4_metric2_f1
        y_<name>_interpolated_model4_metric3_recall
        y_<name>_interpolated_model4_metric4_precision
        y_<name>_interpolated_model4_metric5_auc

PREDICTION_MODE = "all_models"
    - For each target y_<name>, attempts to load, for each enabled model M:
        y_<name>_M.joblib
      and outputs:
        y_<name>                              (actual, once)
        y_<name>_M_interpolated_model4
        y_<name>_M_interpolated_model4_metric1
        y_<name>_M_interpolated_model4_metric2_f1
        y_<name>_M_interpolated_model4_metric3_recall
        y_<name>_M_interpolated_model4_metric4_precision
        y_<name>_M_interpolated_model4_metric5_auc

Global metrics are pulled from Stage-2 CSV (model_cv_results_parallel.csv):

    - For the best model: F1/Recall/Precision/AUC populated.
    - For other models: metrics populated if present in CSV,
      otherwise NaN with a warning.

Inputs
------
- new_data.csv
- feature_importances/feature_importances_<y>.csv
- trained_models/y_<target>_<MODEL>.joblib
- trained_models/y_<target>_best.joblib
- trained_models/model_cv_results_parallel.csv

Output
------
- stage3_predictions.csv with id_pwd_id and all prediction columns.

Logs
----
- logs/y_<target>_stage3.log per target.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, load
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

# Optional MLflow (often off for Stage-3)
try:
    import mlflow

    mlflow_available = True
except Exception:  # pragma: no cover - optional
    mlflow = None  # type: ignore[assignment]
    mlflow_available = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_PATH = "new_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
MODELS_DIR = Path("trained_models")
LOG_DIR = Path("logs")

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

RANDOM_STATE = 42

CHUNK_SIZE = 50_000
CPU_COUNT = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

# ✔ Models we consider for prediction when PREDICTION_MODE="all_models"
ENABLED_MODELS: List[str] = ["RF", "LGBM", "XGB", "HGB", "CB"]

# Prediction mode: "best" or "all_models"
PREDICTION_MODE = "best"

# MLflow (optional)
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "stage3_scoring"

PREDICTIONS_OUTPUT_PATH = Path("stage3_predictions.csv")
CV_RESULTS_CSV = MODELS_DIR / "model_cv_results_parallel.csv"

FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("stage3")


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
            LOG_DIR / f"{y_col}_stage3.log",
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
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_cv_results() -> Optional[pd.DataFrame]:
    """Load Stage-2 CV results if available."""
    if not CV_RESULTS_CSV.exists():
        LOG.warning("Stage-2 CV results not found at %s", CV_RESULTS_CSV)
        return None
    return pd.read_csv(CV_RESULTS_CSV)


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
    logger: logging.Logger,
) -> Optional[List[str]]:
    """Load Stage-1 importance file for y_col and select usable features."""
    path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
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
    """Prepare numeric + CatBoost views for classification predictions."""
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols and y_optional is not None:
        mask = y_optional.notna()
        if mask.any():
            logger.info(
                "Fitting CatBoostEncoder on %d labeled rows for prediction.",
                int(mask.sum()),
            )
            encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
            encoder = encoder.fit(X_num.loc[mask, :], y_optional.loc[mask])
            X_num = encoder.transform(X_num)
        else:
            logger.info(
                "No labeled rows for this target in Stage-3; "
                "dropping object columns for numeric view.",
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
        X_cb[col] = X_cb[col].astype("string").fillna(CAT_FILL_VALUE)

    return X_num, X_cb


def available_classification_models(y_col: str) -> Dict[str, Path]:
    """Return available classification model paths for a target."""
    paths: Dict[str, Path] = {}
    for name in ENABLED_MODELS:
        path = MODELS_DIR / f"{y_col}_{name}.joblib"
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
    """Run Stage-3 scoring for a single target (classification only)."""
    logger = get_target_logger(y_col)
    logger.info(
        "=== Stage-3 scoring for %s (prediction_mode=%s) ===",
        y_col,
        PREDICTION_MODE,
    )

    selected = load_feature_importances_for_target(y_col=y_col, logger=logger)
    if not selected:
        return None

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

    best_model_path = MODELS_DIR / f"{y_col}_best.joblib"
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
        if not model_paths and not best_model_path.exists():
            logger.warning(
                "No individual or best models for %s; skipping.",
                y_col,
            )
            return None

    X_num, X_cb = prepare_views_classification_for_prediction(
        X_full,
        y_optional=y_actual,
        logger=logger,
    )

    n_rows = df.shape[0]
    chunks = list(range(0, n_rows, CHUNK_SIZE))

    base_out = pd.DataFrame(index=df.index)
    for col in id_cols:
        if col in df.columns:
            base_out[col] = df[col]
    if y_actual is not None:
        base_out[y_col] = y_actual

    # ------------------------------------------------------------------ #
    # Mode: BEST
    # ------------------------------------------------------------------ #
    if PREDICTION_MODE == "best":
        model = load(best_model_path)
        all_preds: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        for start in chunks:
            end = min(start + CHUNK_SIZE, n_rows)
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
                    f"Unsupported best model type for {y_col}: {type(model)}",
                )

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

        if cv_row is not None:
            best_name = cv_row.get("best_model")
            if isinstance(best_name, float) and np.isnan(best_name):
                best_name = None
        else:
            best_name = None

        if cv_row is not None and best_name is not None:
            prefix = f"{best_name}_"
            f1_val = cv_row.get(f"{prefix}f1_macro", np.nan)
            rec_val = cv_row.get(f"{prefix}recall_macro", np.nan)
            prec_val = cv_row.get(f"{prefix}precision_macro", np.nan)
            acc_val = cv_row.get(f"{prefix}accuracy", np.nan)
            auc_val = cv_row.get(f"{prefix}auc", np.nan)
        else:
            f1_val = rec_val = prec_val = acc_val = auc_val = np.nan

        out[f"{y_col}_interpolated_model4_metric2_f1"] = f1_val
        out[f"{y_col}_interpolated_model4_metric3_recall"] = rec_val
        out[f"{y_col}_interpolated_model4_metric4_precision"] = prec_val
        out[f"{y_col}_interpolated_model4_metric5_auc"] = auc_val

        logger.info("Finished Stage-3 scoring for %s", y_col)
        return out

    # ------------------------------------------------------------------ #
    # Mode: ALL MODELS
    # ------------------------------------------------------------------ #
    models_available = model_paths
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
            end = min(start + CHUNK_SIZE, n_rows)
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

        if cv_row is not None and name in ["RF", "LGBM", "XGB", "HGB", "CB"]:
            metric_prefix = f"{name}_"
            f1_val = cv_row.get(f"{metric_prefix}f1_macro", np.nan)
            rec_val = cv_row.get(f"{metric_prefix}recall_macro", np.nan)
            prec_val = cv_row.get(f"{metric_prefix}precision_macro", np.nan)
            acc_val = cv_row.get(f"{metric_prefix}accuracy", np.nan)
            auc_val = cv_row.get(f"{metric_prefix}auc", np.nan)
        else:
            f1_val = rec_val = prec_val = acc_val = np.nan

        out[f"{prefix_cols}_metric2_f1"] = f1_val
        out[f"{prefix_cols}_metric3_recall"] = rec_val
        out[f"{prefix_cols}_metric4_precision"] = prec_val
        out[f"{prefix_cols}_metric5_auc"] = auc_val

    logger.info("Finished Stage-3 scoring for %s", y_col)
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage-3 predictions over all y_* targets on new_data.csv."""
    LOG.info("Loading new data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    id_cols, ft_cols, _ = detect_columns(df)
    LOG.info(
        "Detected %d id_, %d ft_ columns in new_data.",
        len(id_cols),
        len(ft_cols),
    )

    df_cv = load_cv_results()

    if USE_MLFLOW and mlflow_available and mlflow is not None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model_best_files = list(MODELS_DIR.glob("y_*_best.joblib"))
    targets_from_models = sorted(
        {p.name.split("_best.joblib")[0] for p in model_best_files},
    )

    if not targets_from_models:
        LOG.error("No y_*_best.joblib models found in %s; aborting Stage-3.", MODELS_DIR)
        return

    LOG.info(
        "Stage-3 will score %d targets with n_jobs_targets=%d (mode=%s).",
        len(targets_from_models),
        N_JOBS_TARGETS,
        PREDICTION_MODE,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
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
        LOG.error("No predictions generated in Stage-3.")
        return

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(
            frame.drop(columns=id_cols, errors="ignore"),
            how="outer",
        )

    combined.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    LOG.info("Stage-3 predictions saved to %s", PREDICTIONS_OUTPUT_PATH)


if __name__ == "__main__":
    main()
