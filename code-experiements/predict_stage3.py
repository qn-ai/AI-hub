#!/usr/bin/env python3
"""
Stage-3: Parallel, Chunked Prediction on a New Dataset with BEST or ALL-MODEL mode.

This script loads trained models from Stage-2 and performs predictions on a
new dataset, optionally logging to MLflow and writing rich logs for each target.

Modes
-----
PREDICTION_MODE = "best"
    For each target y_<name>, load:
        trained_models/y_<name>_best.joblib
    Produces columns:
        y_<name>
        y_<name>_interpolated_model4
        y_<name>_interpolated_model4_metric1       (probability)
        y_<name>_interpolated_model4_metric2_f1
        y_<name>_interpolated_model4_metric3_recall
        y_<name>_interpolated_model4_metric4_precision
        y_<name>_interpolated_model4_metric5_auc

PREDICTION_MODE = "all_models"
    Attempts to load:
        y_<name>_RF.joblib
        y_<name>_LGBM.joblib
        y_<name>_XGB.joblib
        y_<name>_HGB.joblib
        y_<name>_CB.joblib
    Produces, for each available model M:
        y_<name>_M_interpolated_model4
        y_<name>_M_interpolated_model4_metric1
        ...
        y_<name>_M_interpolated_model4_metric5_auc

Inputs
------
- new_data.csv
- feature_importances/<feature_importances_y>.csv
- trained_models/y_<target>_<MODEL>.joblib
- model_cv_results_parallel.csv (Stage-2 metrics, best model only)

Outputs
-------
- stage3_predictions.csv
- logs/y_<target>_stage3.log
- Optional: MLflow logs (disabled by default)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
from joblib import Parallel, delayed

# Optional MLflow
try:
    import mlflow  # noqa: F401
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

NEW_DATA_PATH = "new_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
TRAINED_MODELS_DIR = Path("trained_models")
STAGE2_METRICS_PATH = "model_cv_results_parallel.csv"
LOG_DIR = Path("logs")

OUTPUT_PATH = "stage3_predictions.csv"
ID_COL = "id_pwd_id"

FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

ROW_CHUNK_SIZE = 50_000
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

# MODE: "best" or "all_models"
PREDICTION_MODE = "best"  # change to "all_models" to enable all models

MODEL_SUFFIXES = ["RF", "LGBM", "XGB", "HGB", "CB"]

# MLflow toggle
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "Stage3_Predictions"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage3")


def get_target_logger(y_col: str) -> logging.Logger:
    """Create a per-target log file under logs/."""
    logger = logging.getLogger(f"stage3.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage3_file", False)
        for h in logger.handlers
    )
    if not exists:
        fh = logging.FileHandler(LOG_DIR / f"{y_col}_stage3.log", "w", encoding="utf-8")
        fh._stage3_file = True
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    logger.propagate = True
    return logger


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Identify id_, ft_, y_ columns by prefix."""
    id_cols = [c for c in df.columns if c.startswith("id_")]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def list_targets_from_models() -> list[str]:
    """Return base target names that exist in trained_models folder."""
    names: list[str] = []
    for path in TRAINED_MODELS_DIR.glob("y_*.joblib"):
        stem = path.stem
        if stem.endswith("_best"):
            names.append(stem[:-5])
        else:
            for suf in MODEL_SUFFIXES:
                ending = f"_{suf}"
                if stem.endswith(ending):
                    names.append(stem[: -len(ending)])
                    break
    return sorted(set(names))


def load_feature_list(y_col: str, logger: logging.Logger) -> list[str]:
    """Load selected features where RF/LGBM/CB/XGB/HGB > 0."""
    fp = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
    if not fp.exists():
        logger.warning("Feature importance file missing: %s", fp)
        return []

    df_imp = pd.read_csv(fp)
    if "feature_name" not in df_imp.columns:
        df_imp = df_imp.rename(columns={df_imp.columns[0]: "feature_name"})

    required = ["RF", "LGBM", "CB", "XGB", "HGB"]
    if any(c not in df_imp.columns for c in required):
        logger.warning("Importance file missing required model columns: %s", fp)
        return []

    mask = (df_imp[required] > 0).all(axis=1)
    return df_imp.loc[mask, "feature_name"].astype(str).tolist()


def numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric view using CatBoostEncoder for tree models."""
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=42)
        dummy = np.zeros(len(X2))
        X2 = enc.fit_transform(X2, dummy)
    elif cat_cols:
        X2 = X2.drop(columns=cat_cols)

    return X2.apply(pd.to_numeric, errors="coerce")


def catboost_view(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical view for CatBoost."""
    X2 = X.copy()
    for col in X2.select_dtypes(include=["object"]).columns:
        X2[col] = X2[col].astype("string").fillna(CAT_FILL_VALUE)
    return X2


def compute_probability(model, Xc: pd.DataFrame) -> np.ndarray:
    """Return probability estimates for models that provide predict_proba."""
    if not hasattr(model, "predict_proba"):
        return np.full(Xc.shape[0], np.nan)

    proba = model.predict_proba(Xc)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 2:
        return proba[:, 1]

    preds = model.predict(Xc)
    return np.array([proba[i, int(cls)] for i, cls in enumerate(preds)])


def load_metrics_stage2() -> Optional[pd.DataFrame]:
    """Load Stage-2 metrics, if available."""
    p = Path(STAGE2_METRICS_PATH)
    if not p.exists():
        log.warning("Missing Stage-2 metrics file: %s", p)
        return None
    return pd.read_csv(p)


def stage2_metrics_for_model(
    dfm: Optional[pd.DataFrame], y_col: str, model_name: str, logger: logging.Logger
) -> dict[str, float]:
    """
    Stage-2 file only stores the *best model* metrics.
    So:
    - if model_name == best_model: return metrics
    - else: return NaN for that model.
    """
    if dfm is None:
        return {"f1": np.nan, "recall": np.nan, "precision": np.nan, "auc": np.nan}

    row = dfm.loc[dfm["target"] == y_col]
    if row.empty:
        return {"f1": np.nan, "recall": np.nan, "precision": np.nan, "auc": np.nan}

    r = row.iloc[0]
    if str(r["best_model"]) != model_name:
        logger.debug("No per-model CV metrics for %s (%s), filling NaN.", y_col, model_name)
        return {"f1": np.nan, "recall": np.nan, "precision": np.nan, "auc": np.nan}

    return {
        "f1": float(r.get("f1", np.nan)),
        "recall": float(r.get("recall", np.nan)),
        "precision": float(r.get("precision", np.nan)),
        "auc": float(r.get("auc", np.nan)),
    }


# ---------------------------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------------------------

def predict_best_model(
    y_col: str,
    df: pd.DataFrame,
    df_metrics: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Predict using only y_<target>_best.joblib."""
    logger = get_target_logger(y_col)

    model_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    if not model_path.exists():
        logger.warning("Missing best model for %s: %s", y_col, model_path)
        return None

    model = joblib.load(model_path)
    logger.info("Loaded best model type: %s", model.__class__.__name__)

    # Actual labels
    y_actual = df.get(y_col, pd.Series(np.nan, index=df.index))

    features = load_feature_list(y_col, logger)
    if not features:
        return None

    X = df[features].copy()

    Xp = catboost_view(X) if isinstance(model, CatBoostClassifier) else numeric_view(X)

    preds = np.empty(len(df), dtype=object)
    probas = np.empty(len(df), dtype=float)

    for start in range(0, len(df), ROW_CHUNK_SIZE):
        end = min(start + ROW_CHUNK_SIZE, len(df))
        logger.info("BEST model: rows [%d:%d) for %s", start, end, y_col)
        Xc = Xp.iloc[start:end]
        preds[start:end] = model.predict(Xc)
        probas[start:end] = compute_probability(model, Xc)

    m = stage2_metrics_for_model(df_metrics, y_col, model_name="", logger=logger)

    base = f"{y_col}_interpolated_model4"

    out = pd.DataFrame(index=df.index)
    out[y_col] = y_actual
    out[base] = preds
    out[f"{base}_metric1"] = probas
    out[f"{base}_metric2_f1"] = m["f1"]
    out[f"{base}_metric3_recall"] = m["recall"]
    out[f"{base}_metric4_precision"] = m["precision"]
    out[f"{base}_metric5_auc"] = m["auc"]

    return out


def predict_all_models(
    y_col: str,
    df: pd.DataFrame,
    df_metrics: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Predict using all available model files for this target."""
    logger = get_target_logger(y_col)

    y_actual = df.get(y_col, pd.Series(np.nan, index=df.index))
    features = load_feature_list(y_col, logger)
    if not features:
        return None

    X = df[features].copy()
    X_num = numeric_view(X)
    X_cb = catboost_view(X)

    out = pd.DataFrame(index=df.index)
    out[y_col] = y_actual

    for suf in MODEL_SUFFIXES:
        path = TRAINED_MODELS_DIR / f"{y_col}_{suf}.joblib"
        if not path.exists():
            logger.warning("%s model missing for %s", suf, y_col)
            continue

        model = joblib.load(path)
        logger.info("Predicting with %s for target %s", suf, y_col)

        if isinstance(model, CatBoostClassifier):
            Xp = X_cb
        else:
            Xp = X_num

        preds = np.empty(len(df), dtype=object)
        probas = np.empty(len(df), dtype=float)

        for start in range(0, len(df), ROW_CHUNK_SIZE):
            end = min(start + ROW_CHUNK_SIZE, len(df))
            logger.info("%s: rows [%d:%d) for %s", suf, start, end, y_col)
            Xc = Xp.iloc[start:end]
            preds[start:end] = model.predict(Xc)
            probas[start:end] = compute_probability(model, Xc)

        m = stage2_metrics_for_model(df_metrics, y_col, suf, logger)

        base = f"{y_col}_{suf}_interpolated_model4"
        out[base] = preds
        out[f"{base}_metric1"] = probas
        out[f"{base}_metric2_f1"] = m["f1"]
        out[f"{base}_metric3_recall"] = m["recall"]
        out[f"{base}_metric4_precision"] = m["precision"]
        out[f"{base}_metric5_auc"] = m["auc"]

    if out.shape[1] <= 1:
        logger.warning("No model outputs produced for %s", y_col)
        return None
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for Stage-3 prediction."""
    log.info("Loading new data: %s", NEW_DATA_PATH)
    df = pd.read_csv(NEW_DATA_PATH, low_memory=False)

    if ID_COL not in df.columns:
        raise ValueError(f"ID column {ID_COL} missing from new data.")

    df_metrics = load_metrics_stage2()
    targets = list_targets_from_models()

    log.info("Targets with trained models: %d", len(targets))
    log.info("Prediction mode: %s", PREDICTION_MODE)

    if USE_MLFLOW and MLFLOW_AVAILABLE:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run(run_name="Stage3_Predictions")
        mlflow.log_param("prediction_mode", PREDICTION_MODE)

    output = df[[ID_COL]].copy()

    predict_fn = predict_best_model if PREDICTION_MODE == "best" else predict_all_models

    results = Parallel(n_jobs=N_JOBS)(
        delayed(predict_fn)(y_col, df, df_metrics) for y_col in targets
    )

    for res in results:
        if res is not None:
            output = output.join(res)

    output.to_csv(OUTPUT_PATH, index=False)
    log.info("Stage-3 output saved to %s", OUTPUT_PATH)

    if USE_MLFLOW and MLFLOW_AVAILABLE:
        mlflow.log_artifact(OUTPUT_PATH)
        mlflow.end_run()


if __name__ == "__main__":
    main()
