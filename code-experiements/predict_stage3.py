#!/usr/bin/env python
"""
Stage-3: Parallel, Chunked Prediction on a New Dataset.

For each target y_<name> that has a best model in trained_models/:

- Loads best model: trained_models/y_<name>_best.joblib
- Loads selected features from: feature_importances/feature_importances_<name>.csv
- Prepares features similar to Stage-2:
    * CatBoost models: raw string categoricals with NA_CAT
    * RF/LGBM/XGB/HGB: numeric view via CatBoostEncoder (unsupervised)

- Applies model to new_data.csv in row chunks (e.g. 50k rows)
- Reuses Stage-2 CV metrics (f1, recall, precision, auc) as global confidence metrics
- Writes predictions to stage3_predictions.csv

Outputs per target:
- y_<target>                         (actual label if present in new_data)
- y_<target>_interpolated_model4     (predicted class)
- y_<target>_interpolated_model4_metric1         (per-row probability / confidence)
- y_<target>_interpolated_model4_metric2_f1      (global F1 from Stage-2)
- y_<target>_interpolated_model4_metric3_recall  (global Recall)
- y_<target>_interpolated_model4_metric4_precision (global Precision)
- y_<target>_interpolated_model4_metric5_auc     (global AUC)

Per-target logs written to:
- logs/y_<target>_stage3.log
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from joblib import Parallel, delayed

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

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
_CPU = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(_CPU - 1, 12), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage3")


def get_target_logger(y_col: str) -> logging.Logger:
    """Return a logger that writes to logs/y_<target>_stage3.log."""
    logger = logging.getLogger(f"stage3.{y_col}")
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage3_file", False)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(LOG_DIR / f"{y_col}_stage3.log", mode="w", encoding="utf-8")
        fh._stage3_file = True  # type: ignore[attr-defined]
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = True
    return logger


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def detect_columns(df: pd.DataFrame):
    id_cols = [c for c in df.columns if c.startswith("id_")]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def list_targets_from_models() -> List[str]:
    """List all targets that actually have a trained best model."""
    targets: List[str] = []
    for p in TRAINED_MODELS_DIR.glob("y_*_best.joblib"):
        name = p.name
        if not name.endswith("_best.joblib"):
            continue
        y_col = name.replace("_best.joblib", "")
        targets.append(y_col)
    return sorted(set(targets))


def load_selected_features(y_col: str, logger: logging.Logger) -> List[str]:
    path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
    if not path.exists():
        logger.warning("No feature_importances file found for %s at %s", y_col, path)
        return []

    df_imp = pd.read_csv(path)
    if "feature_name" not in df_imp.columns:
        df_imp = df_imp.rename(columns={df_imp.columns[0]: "feature_name"})

    required = ["RF", "LGBM", "CB", "XGB", "HGB"]
    missing = [c for c in required if c not in df_imp.columns]
    if missing:
        logger.warning(
            "Importance file %s for %s missing %s", path, y_col, missing
        )
        return []

    mask = (df_imp[required] > 0).all(axis=1)
    sel = df_imp.loc[mask, "feature_name"].astype(str).tolist()
    return sel


def prepare_catboost_view(X: pd.DataFrame) -> pd.DataFrame:
    X_cb = X.copy()
    cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype("string").fillna(CAT_FILL_VALUE)
    return X_cb


def prepare_numeric_view_with_encoder(X: pd.DataFrame) -> pd.DataFrame:
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=42)
        dummy = np.zeros(len(X_num))
        X_num = enc.fit_transform(X_num, dummy)
    elif not USE_CATBOOST_ENCODER and cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    return X_num


def compute_probs(model, X_chunk: pd.DataFrame) -> np.ndarray:
    """Return per-row probability (confidence)."""
    if not hasattr(model, "predict_proba"):
        return np.full(X_chunk.shape[0], np.nan)

    proba = model.predict_proba(X_chunk)
    if proba.ndim == 1:
        return proba

    if proba.shape[1] == 2:
        return proba[:, 1]

    preds = model.predict(X_chunk)
    out = np.empty(X_chunk.shape[0], dtype=float)
    for i, cls in enumerate(preds):
        out[i] = proba[i, cls]
    return out


def get_stage2_metrics_for_target(
    df_metrics: Optional[pd.DataFrame],
    y_col: str,
) -> dict:
    if df_metrics is None:
        return {"f1": np.nan, "recall": np.nan, "precision": np.nan, "auc": np.nan}

    row = df_metrics.loc[df_metrics["target"] == y_col]
    if row.empty:
        return {"f1": np.nan, "recall": np.nan, "precision": np.nan, "auc": np.nan}

    row = row.iloc[0]
    return {
        "f1": float(row.get("f1", np.nan)),
        "recall": float(row.get("recall", np.nan)),
        "precision": float(row.get("precision", np.nan)),
        "auc": float(row.get("auc", np.nan)),
    }


# ---------------------------------------------------------------------
# PER-TARGET PREDICTION
# ---------------------------------------------------------------------

def predict_for_target(
    y_col: str,
    df_new: pd.DataFrame,
    df_metrics: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    logger = get_target_logger(y_col)
    logger.info("=== Stage-3 prediction for %s ===", y_col)

    model_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    if not model_path.exists():
        logger.warning("No model file for %s at %s. Skipping.", y_col, model_path)
        return None

    model = joblib.load(model_path)
    model_name = model.__class__.__name__
    logger.info("Loaded model type: %s", model_name)

    if y_col in df_new.columns:
        y_actual = df_new[y_col]
    else:
        y_actual = pd.Series([np.nan] * len(df_new), index=df_new.index)
        logger.info("No actual %s in new_data; filling NaN.", y_col)

    selected_features = load_selected_features(y_col, logger)
    if not selected_features:
        logger.warning("No selected features for %s. Skipping.", y_col)
        return None

    missing_features = [f for f in selected_features if f not in df_new.columns]
    if missing_features:
        logger.warning(
            "Missing features in new_data for %s: %s. Skipping.",
            y_col,
            missing_features,
        )
        return None

    X = df_new[selected_features].copy()
    n_rows = X.shape[0]
    logger.info("Predicting %d rows with %d features.", n_rows, X.shape[1])

    if model_name == "CatBoostClassifier":
        X_prepared = prepare_catboost_view(X)
    else:
        X_prepared = prepare_numeric_view_with_encoder(X)

    y_pred_all = np.empty(n_rows, dtype=object)
    y_prob_all = np.empty(n_rows, dtype=float)

    for start in range(0, n_rows, ROW_CHUNK_SIZE):
        end = min(start + ROW_CHUNK_SIZE, n_rows)
        logger.info("Processing rows [%d:%d)", start, end)
        X_chunk = X_prepared.iloc[start:end]

        preds_chunk = model.predict(X_chunk)
        probs_chunk = compute_probs(model, X_chunk)

        y_pred_all[start:end] = preds_chunk
        y_prob_all[start:end] = probs_chunk

    m = get_stage2_metrics_for_target(df_metrics, y_col)
    logger.info(
        "Stage-2 metrics reused for %s: F1=%.4f Recall=%.4f Precision=%.4f AUC=%.4f",
        y_col,
        m["f1"],
        m["recall"],
        m["precision"],
        m["auc"],
    )

    res = pd.DataFrame(index=df_new.index)
    res[y_col] = y_actual
    res[f"{y_col}_interpolated_model4"] = y_pred_all
    res[f"{y_col}_interpolated_model4_metric1"] = y_prob_all
    res[f"{y_col}_interpolated_model4_metric2_f1"] = m["f1"]
    res[f"{y_col}_interpolated_model4_metric3_recall"] = m["recall"]
    res[f"{y_col}_interpolated_model4_metric4_precision"] = m["precision"]
    res[f"{y_col}_interpolated_model4_metric5_auc"] = m["auc"]

    logger.info("Completed Stage-3 prediction for %s.", y_col)
    return res


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    log.info("Loading new_data from %s", NEW_DATA_PATH)
    df_new = pd.read_csv(NEW_DATA_PATH, low_memory=False)
    if ID_COL not in df_new.columns:
        raise ValueError(f"ID column '{ID_COL}' not found in new_data.csv.")

    id_cols, ft_cols, y_cols_new = detect_columns(df_new)
    log.info(
        "new_data: %d rows, %d columns, %d y_ columns.",
        df_new.shape[0],
        df_new.shape[1],
        len(y_cols_new),
    )

    # Stage-2 metrics
    if Path(STAGE2_METRICS_PATH).exists():
        df_metrics = pd.read_csv(STAGE2_METRICS_PATH)
        log.info("Loaded Stage-2 metrics from %s", STAGE2_METRICS_PATH)
    else:
        df_metrics = None
        log.warning("No Stage-2 metrics at %s; metric columns will be NaN.", STAGE2_METRICS_PATH)

    targets = list_targets_from_models()
    log.info(
        "Found %d targets with trained models in %s",
        len(targets),
        TRAINED_MODELS_DIR,
    )

    out = df_new[[ID_COL]].copy()

    log.info(
        "Starting Stage-3 prediction (targets=%d, N_JOBS_TARGETS=%d, ROW_CHUNK_SIZE=%d)",
        len(targets),
        N_JOBS_TARGETS,
        ROW_CHUNK_SIZE,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(predict_for_target)(y_col, df_new, df_metrics) for y_col in targets
    )

    for res in results:
        if res is None:
            continue
        out = out.join(res)

    out.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved Stage-3 predictions to %s", OUTPUT_PATH)
    print(f"\n✅ [Stage-3] Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
