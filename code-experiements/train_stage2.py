#!/usr/bin/env python
"""Stage-2: train per-target models using Stage-1 feature_importances
with optional MLflow logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# =====================================================================
# CONFIG
# =====================================================================

DATA_PATH = "input_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
MODELS_DIR = Path("trained_models")

RESULTS_CSV_PATH = "model_cv_results_parallel.csv"
RESULTS_JSON_PATH = "model_cv_results_parallel.json"

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

MIN_SAMPLES_PER_TARGET = 200
N_SPLITS = 3
RANDOM_STATE = 42

_CPU = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(_CPU - 1, 12), 2)

USE_CATBOOST_ENCODER = True

# ---- MLflow toggle ----
USE_MLFLOW_STAGE2 = False
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "stage2_multi_target_training"

if USE_MLFLOW_STAGE2:
    import mlflow  # type: ignore

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
else:
    mlflow = None  # type: ignore

# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# =====================================================================
# HELPERS
# =====================================================================


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_selected_features_for_target(
    y_col: str,
    ft_cols_all: List[str],
) -> List[str]:
    all_csvs = list(FEATURE_IMPORTANCE_DIR.glob("*.csv"))
    if not all_csvs:
        log.error(
            "No feature importance CSV files found in %s",
            FEATURE_IMPORTANCE_DIR,
        )
        return []

    excluded_suffixes = ("_RF.csv", "_LGBM.csv", "_CB.csv", "_XGB.csv", "_HGB.csv")
    combined_files = [p for p in all_csvs if not p.name.endswith(excluded_suffixes)]
    if not combined_files:
        log.error(
            "All feature importance files appear to be per-model only; "
            "no combined files found."
        )
        return []

    matching_files = [p for p in combined_files if y_col in p.name]
    if not matching_files:
        log.error(
            "No combined feature importance file found for target %s "
            "(searched filenames containing '%s')",
            y_col,
            y_col,
        )
        return []

    file_path = max(matching_files, key=lambda p: p.stat().st_size)
    log.info("Using feature importance file for %s: %s", y_col, file_path)

    fi_df = pd.read_csv(file_path)
    required_cols = ["RF", "LGBM", "CB", "XGB", "HGB"]
    missing = [c for c in required_cols if c not in fi_df.columns]
    if missing:
        log.error(
            "Importance file %s for %s is missing required columns %s",
            file_path,
            y_col,
            missing,
        )
        return []

    if "feature_name" not in fi_df.columns:
        log.warning(
            "Column 'feature_name' not found in %s; using first column as feature_name",
            file_path,
        )
        fi_df = fi_df.rename(columns={fi_df.columns[0]: "feature_name"})

    mask = (fi_df[required_cols] > 0).all(axis=1)
    selected = fi_df.loc[mask, "feature_name"].astype(str).tolist()
    selected = [f for f in selected if f in ft_cols_all]

    log.info(
        "Target %s: %d features selected from %s (RF/LGBM/CB/XGB/HGB > 0)",
        y_col,
        len(selected),
        file_path.name,
    )
    return selected


def encode_for_non_catboost(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER:
        if cat_cols:
            enc = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
            X2 = enc.fit_transform(X, y)
        else:
            X2 = X.copy()
    else:
        if cat_cols:
            log.warning(
                "USE_CATBOOST_ENCODER=False -> dropping %d object columns: %s",
                len(cat_cols),
                cat_cols,
            )
        X2 = X.drop(columns=cat_cols)

    return X2.apply(pd.to_numeric, errors="coerce")


def prepare_catboost_input(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str]]:
    X2 = X.copy()
    cat_cols = X2.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X2[c] = X2[c].astype("string").fillna("NA_CAT")
    return X2, cat_cols


def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    is_binary: bool,
    n_splits: int = N_SPLITS,
) -> dict:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }

    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(Xtr, ytr)
        pred = model.predict(Xv)

        metrics["accuracy"].append(accuracy_score(yv, pred))
        metrics["precision"].append(
            precision_score(yv, pred, average="binary" if is_binary else "macro")
        )
        metrics["recall"].append(
            recall_score(yv, pred, average="binary" if is_binary else "macro")
        )
        metrics["f1"].append(
            f1_score(yv, pred, average="binary" if is_binary else "macro")
        )

        try:
            proba = model.predict_proba(Xv)
            if is_binary:
                auc = roc_auc_score(yv, proba[:, 1])
            else:
                auc = roc_auc_score(yv, proba, multi_class="ovr")
        except Exception:
            auc = float("nan")

        metrics["auc"].append(auc)

    return {k: float(np.nanmean(v)) for k, v in metrics.items()}


# =====================================================================
# PER-TARGET TRAINING
# =====================================================================


def train_one_target(y_col: str) -> Optional[dict]:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    id_cols, ft_cols, y_cols = detect_columns(df)

    if y_col not in y_cols:
        log.warning("Column %s not recognised as y_ target. Skipping.", y_col)
        return None

    df_t = df[df[y_col].notna()].copy()
    n_rows = len(df_t)
    if n_rows < MIN_SAMPLES_PER_TARGET:
        log.info(
            "Skipping %s: only %d non-missing rows (< %d)",
            y_col,
            n_rows,
            MIN_SAMPLES_PER_TARGET,
        )
        return None

    y = df_t[y_col].astype("category").cat.codes
    n_classes = y.nunique()
    if n_classes < 2:
        log.info("Skipping %s: only one class after encoding.", y_col)
        return None
    is_binary = n_classes == 2

    selected_features = load_selected_features_for_target(y_col, ft_cols)
    if not selected_features:
        log.info("No selected features for %s. Skipping.", y_col)
        return None

    X = df_t[selected_features]
    X_enc = encode_for_non_catboost(X, y)
    X_cb, cat_cols_cb = prepare_catboost_input(X)

    start_time = time.time()

    # MLflow context (no-op if disabled)
    if USE_MLFLOW_STAGE2:
        run_ctx = mlflow.start_run(run_name=y_col)  # type: ignore
    else:
        run_ctx = nullcontext()

    with run_ctx:
        if USE_MLFLOW_STAGE2:
            mlflow.log_param("target", y_col)  # type: ignore
            mlflow.log_param("n_samples", int(n_rows))  # type: ignore
            mlflow.log_param("n_features", len(selected_features))  # type: ignore
            mlflow.log_param("n_classes", int(n_classes))  # type: ignore
            mlflow.log_param("is_binary", bool(is_binary))  # type: ignore

        results: dict[str, dict] = {}
        best_f1 = -1.0
        best_model = None
        best_name = None

        # RF
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        res_rf = evaluate_model_cv(rf, X_enc, y, is_binary)
        results["RF"] = res_rf
        if USE_MLFLOW_STAGE2:
            for m_name, m_val in res_rf.items():
                mlflow.log_metric(f"RF_{m_name}", m_val)  # type: ignore
        if res_rf["f1"] > best_f1:
            best_f1 = res_rf["f1"]
            best_model = rf
            best_name = "RF"

        # LGBM
        lgbm = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="binary" if is_binary else "multiclass",
            num_class=n_classes if not is_binary else None,
        )
        res_lgb = evaluate_model_cv(lgbm, X_enc, y, is_binary)
        results["LGBM"] = res_lgb
        if USE_MLFLOW_STAGE2:
            for m_name, m_val in res_lgb.items():
                mlflow.log_metric(f"LGBM_{m_name}", m_val)  # type: ignore
        if res_lgb["f1"] > best_f1:
            best_f1 = res_lgb["f1"]
            best_model = lgbm
            best_name = "LGBM"

        # XGB
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="binary:logistic" if is_binary else "multi:softprob",
            num_class=n_classes if not is_binary else None,
        )
        res_xgb = evaluate_model_cv(xgb, X_enc, y, is_binary)
        results["XGB"] = res_xgb
        if USE_MLFLOW_STAGE2:
            for m_name, m_val in res_xgb.items():
                mlflow.log_metric(f"XGB_{m_name}", m_val)  # type: ignore
        if res_xgb["f1"] > best_f1:
            best_f1 = res_xgb["f1"]
            best_model = xgb
            best_name = "XGB"

        # HGB
        hgb = HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=None,
            learning_rate=0.05,
            max_bins=255,
            l2_regularization=0.0,
            random_state=RANDOM_STATE,
        )
        res_hgb = evaluate_model_cv(hgb, X_enc, y, is_binary)
        results["HGB"] = res_hgb
        if USE_MLFLOW_STAGE2:
            for m_name, m_val in res_hgb.items():
                mlflow.log_metric(f"HGB_{m_name}", m_val)  # type: ignore
        if res_hgb["f1"] > best_f1:
            best_f1 = res_hgb["f1"]
            best_model = hgb
            best_name = "HGB"

        # CatBoost
        cb = CatBoostClassifier(
            iterations=400,
            depth=7,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            loss_function="Logloss" if is_binary else "MultiClass",
            verbose=False,
        )
        res_cb = evaluate_model_cv(cb, X_cb, y, is_binary)
        results["CB"] = res_cb
        if USE_MLFLOW_STAGE2:
            for m_name, m_val in res_cb.items():
                mlflow.log_metric(f"CB_{m_name}", m_val)  # type: ignore
        if res_cb["f1"] > best_f1:
            best_f1 = res_cb["f1"]
            best_model = cb
            best_name = "CB"

        if best_model is None:
            log.warning("No valid model for %s.", y_col)
            if USE_MLFLOW_STAGE2:
                mlflow.log_param("best_model", "None")  # type: ignore
            return None

        elapsed = time.time() - start_time

        # Retrain best model on full data
        if best_name == "CB":
            best_model.fit(X_cb, y)
        else:
            best_model.fit(X_enc, y)

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{y_col}_best.joblib"
        joblib.dump(best_model, model_path)

        if USE_MLFLOW_STAGE2:
            mlflow.log_param("best_model", best_name)  # type: ignore
            mlflow.log_metric("best_f1", best_f1)  # type: ignore
            mlflow.log_metric("time_sec", elapsed)  # type: ignore
            mlflow.log_artifact(str(model_path))  # type: ignore

        log.info(
            "Finished %s -> best=%s, f1=%.4f, time=%.1fs, features=%d, samples=%d",
            y_col,
            best_name,
            best_f1,
            elapsed,
            len(selected_features),
            n_rows,
        )

    return {
        "target": y_col,
        "best_model": best_name,
        "metrics": results,
        "time_sec": elapsed,
        "n_features": len(selected_features),
        "n_samples": int(n_rows),
    }


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    _, _, y_cols = detect_columns(df)

    log.info(
        "Stage-2: training models for %d targets using %d workers",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(train_one_target)(y_col) for y_col in y_cols
    )

    results = [r for r in results if r is not None]
    if not results:
        log.warning("No targets were successfully trained.")
        return

    rows = []
    for r in results:
        best_name = r["best_model"]
        best_metrics = r["metrics"][best_name]
        row = {
            "target": r["target"],
            "best_model": best_name,
            "time_sec": r["time_sec"],
            "n_features": r["n_features"],
            "n_samples": r["n_samples"],
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1": best_metrics["f1"],
            "auc": best_metrics["auc"],
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(RESULTS_CSV_PATH, index=False)
    log.info("Saved summary CSV → %s", RESULTS_CSV_PATH)

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved full results JSON → %s", RESULTS_JSON_PATH)


if __name__ == "__main__":
    main()
