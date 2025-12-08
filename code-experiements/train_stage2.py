#!/usr/bin/env python
"""
Stage-2: Per-Target Model Training with Dynamic CV Folds.

This stage trains a best model per target using features selected in Stage-1.
It handles:
- targets with only one label (skipped as non-learnable),
- very imbalanced targets (dynamic n_splits based on min class count),
- mixed feature types (numeric + categorical),
- missing values (NaN-aware models).

Inputs
------
- input_data.csv :
    Wide table with:
        - id_* identifier columns
        - ft_* feature columns (numeric or object)
        - y_* target columns (binary or multiclass)

- feature_importances/feature_importances_<y>.csv :
    From Stage-1, must contain:
        feature_name, RF, LGBM, CB, XGB, HGB, mean_rank

Outputs
-------
- trained_models/y_<target>_best.joblib
- model_cv_results_parallel.csv
- model_cv_results_parallel.json
- logs/y_<target>_stage2.log
- skipped_targets_stage2.csv (targets skipped with a reason)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_PATH = "input_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
TRAINED_MODELS_DIR = Path("trained_models")
LOG_DIR = Path("logs")
RESULTS_CSV = "model_cv_results_parallel.csv"
RESULTS_JSON = "model_cv_results_parallel.json"
SKIPPED_CSV = "skipped_targets_stage2.csv"

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

RANDOM_STATE = 42

# CV folds: per target n_splits = min(MAX_N_SPLITS, min_class_count)
MAX_N_SPLITS = 5

# Parallelism
_CPU = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(_CPU - 1, 12), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage2")


def get_target_logger(y_col: str) -> logging.Logger:
    """Return per-target logger writing to logs/y_<target>_stage2.log."""
    logger = logging.getLogger(f"stage2.{y_col}")
    logger.setLevel(logging.INFO)

    # Attach file handler once
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage2_file", False)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(LOG_DIR / f"{y_col}_stage2.log", mode="w", encoding="utf-8")
        fh._stage2_file = True  # type: ignore[attr-defined]
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = True
    return logger


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Detect id_, ft_, y_ columns by prefix."""
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_selected_features(y_col: str, logger: logging.Logger) -> List[str]:
    """Load selected features for a target from Stage-1 combined CSV.

    Only keep rows where RF, LGBM, CB, XGB, HGB > 0.
    """
    path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
    if not path.exists():
        logger.warning("Feature importance file not found: %s", path)
        return []

    df_imp = pd.read_csv(path)
    if "feature_name" not in df_imp.columns:
        df_imp = df_imp.rename(columns={df_imp.columns[0]: "feature_name"})

    required = ["RF", "LGBM", "CB", "XGB", "HGB"]
    missing = [c for c in required if c not in df_imp.columns]
    if missing:
        logger.warning("Importance file %s missing columns %s", path, missing)
        return []

    mask = (df_imp[required] > 0).all(axis=1)
    sel = df_imp.loc[mask, "feature_name"].astype(str).tolist()
    logger.info("Selected %d features for %s", len(sel), y_col)
    return sel


def prepare_target_and_cv(
    y_raw: pd.Series,
    max_splits: int,
    logger: logging.Logger,
) -> Tuple[Optional[pd.Series], Optional[LabelEncoder], Optional[int]]:
    """Encode target and choose n_splits based on class distribution.

    - Convert to string categories
    - Compute class counts
    - min_count = min(counts)
    - If min_count < 2 → target is not learnable → return (None, None, None)
    - Else n_splits_target = min(max_splits, min_count)
    - Encode with LabelEncoder -> contiguous 0..K-1

    Returns:
        y_enc, label_encoder, n_splits_target
    """
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    logger.info("Class distribution: %s", counts.to_dict())

    min_count = int(counts.min())
    n_classes = counts.shape[0]

    if min_count < 2:
        logger.warning(
            "Skipping target: min class count = %d < 2 (n_classes=%d).",
            min_count,
            n_classes,
        )
        return None, None, None

    n_splits_target = int(min(max_splits, min_count))
    if n_splits_target < 2:
        logger.warning("Skipping target: n_splits would be %d < 2.", n_splits_target)
        return None, None, None

    le = LabelEncoder()
    y_enc = pd.Series(
        le.fit_transform(y_str),
        index=y_raw.index,
        dtype="int64",
    )

    logger.info(
        "Using n_splits=%d (min_class_count=%d, n_classes=%d, labels=%s)",
        n_splits_target,
        min_count,
        n_classes,
        np.unique(y_enc).tolist(),
    )

    return y_enc, le, n_splits_target


def prepare_catboost_view(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare CatBoost view: string categoricals, NaN -> CAT_FILL_VALUE."""
    X_cb = X.copy()
    cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype("string").fillna(CAT_FILL_VALUE)
    return X_cb, cat_cols


def prepare_numeric_view_with_encoder(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare numeric encoded view using CatBoostEncoder on all rows.

    This is an unsupervised encoding (dummy target), but consistent with Stage-3.
    """
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        dummy_y = np.zeros(len(X_num))
        X_num = enc.fit_transform(X_num, dummy_y)
    elif not USE_CATBOOST_ENCODER and cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    return X_num


def build_models(is_binary: bool) -> Dict[str, object]:
    """Return dict of model_name -> estimator prototype."""
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            objective=lgbm_obj,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGB": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=xgb_obj,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HGB": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "CB": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function=cb_loss,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
    }
    return models


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    is_binary: bool,
) -> Dict[str, float]:
    """Compute metrics for a single fold."""
    avg = "binary" if is_binary else "macro"
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)

    if y_proba is not None:
        try:
            if is_binary:
                # y_proba is shape (n_samples,) or (n_samples,2)
                if y_proba.ndim == 1:
                    metrics["auc"] = roc_auc_score(y_true, y_proba)
                else:
                    metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["auc"] = roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                )
        except Exception:
            metrics["auc"] = np.nan
    else:
        metrics["auc"] = np.nan

    return metrics


def cross_validate_target(
    y_col: str,
    X: pd.DataFrame,
    y_enc: pd.Series,
    n_splits_target: int,
    logger: logging.Logger,
) -> Tuple[Optional[str], Dict[str, Dict[str, float]], Optional[object]]:
    """Cross-validate all models for a target and return best model name + metrics + best estimator."""
    is_binary = y_enc.nunique() == 2
    logger.info("Target %s is %s", y_col, "binary" if is_binary else "multiclass")

    cv = StratifiedKFold(
        n_splits=n_splits_target,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Prepare views
    X_num = prepare_numeric_view_with_encoder(X)
    X_cb, cat_cols = prepare_catboost_view(X)
    cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

    models = build_models(is_binary)
    all_results: Dict[str, Dict[str, float]] = {}
    best_model_name: Optional[str] = None
    best_f1: float = -np.inf
    best_estimator: Optional[object] = None

    for name, model_proto in models.items():
        logger.info("CV for model: %s", name)
        metrics_list: List[Dict[str, float]] = []

        # Clone to avoid cross-fold contamination
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_num, y_enc), start=1):
            logger.info("  Fold %d/%d", fold_idx, n_splits_target)
            if name == "CB":
                X_train = X_cb.iloc[train_idx]
                X_val = X_cb.iloc[val_idx]
                y_train = y_enc.iloc[train_idx]
                y_val = y_enc.iloc[val_idx]

                model = model_proto.__class__(**model_proto.get_params())
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_indices if cat_indices else None,
                )
                y_pred = model.predict(X_val)
                y_proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_val)
                    if is_binary and proba.shape[1] == 2:
                        y_proba = proba[:, 1]
                    else:
                        y_proba = proba
            else:
                X_train = X_num.iloc[train_idx]
                X_val = X_num.iloc[val_idx]
                y_train = y_enc.iloc[train_idx]
                y_val = y_enc.iloc[val_idx]

                model = model_proto.__class__(**model_proto.get_params())
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_val)
                    if is_binary and proba.shape[1] == 2:
                        y_proba = proba[:, 1]
                    else:
                        y_proba = proba

            m = compute_metrics(
                y_true=y_val.to_numpy(),
                y_pred=y_pred,
                y_proba=y_proba,
                is_binary=is_binary,
            )
            metrics_list.append(m)

        # Aggregate metrics across folds
        agg = {
            key: float(np.nanmean([m[key] for m in metrics_list]))
            for key in metrics_list[0].keys()
        }
        agg["n_splits"] = float(n_splits_target)
        all_results[name] = agg
        logger.info("  CV results for %s: %s", name, agg)

        if agg["f1"] > best_f1:
            best_f1 = agg["f1"]
            best_model_name = name

    if best_model_name is None:
        logger.warning("No valid model for target %s", y_col)
        return None, all_results, None

    logger.info("Best model for %s is %s (F1=%.4f)", y_col, best_model_name, best_f1)

    # Refit best model on full data for this target
    final_models = build_models(is_binary)
    best_model_proto = final_models[best_model_name]

    if best_model_name == "CB":
        best_model = best_model_proto.__class__(**best_model_proto.get_params())
        best_model.fit(X_cb, y_enc, cat_features=cat_indices if cat_indices else None)
    else:
        X_num_full = prepare_numeric_view_with_encoder(X)
        best_model = best_model_proto.__class__(**best_model_proto.get_params())
        best_model.fit(X_num_full, y_enc)

    return best_model_name, all_results, best_model


# ---------------------------------------------------------------------
# PER-TARGET PIPELINE (RUN IN PARALLEL)
# ---------------------------------------------------------------------

def process_target(y_col: str, df: pd.DataFrame) -> Dict:
    """Train and select best model for a single target."""
    logger = get_target_logger(y_col)
    logger.info("=== Stage-2 training for %s ===", y_col)

    df_target = df[df[y_col].notna()].copy()
    n_rows = df_target.shape[0]
    if n_rows < 2:
        logger.warning("Skipping %s: <2 labelled rows (%d).", y_col, n_rows)
        return {"target": y_col, "skipped": True, "reason": "too_few_rows"}

    # Load selected features
    selected_features = load_selected_features(y_col, logger)
    if not selected_features:
        logger.warning("Skipping %s: no selected features.", y_col)
        return {"target": y_col, "skipped": True, "reason": "no_selected_features"}

    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        logger.warning(
            "Skipping %s: missing features in data: %s", y_col, missing_features
        )
        return {"target": y_col, "skipped": True, "reason": "missing_features"}

    X = df_target[selected_features].copy()
    y_raw = df_target[y_col]

    # Prepare target + dynamic n_splits
    y_enc, le, n_splits_target = prepare_target_and_cv(
        y_raw=y_raw,
        max_splits=MAX_N_SPLITS,
        logger=logger,
    )
    if y_enc is None:
        return {"target": y_col, "skipped": True, "reason": "too_few_classes"}

    best_model_name, all_results, best_model = cross_validate_target(
        y_col=y_col,
        X=X,
        y_enc=y_enc,
        n_splits_target=n_splits_target,
        logger=logger,
    )

    if best_model is None or best_model_name is None:
        return {"target": y_col, "skipped": True, "reason": "no_valid_model"}

    # Save best model
    model_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Saved best model for %s to %s", y_col, model_path)

    # Flatten metrics for CSV/JSON
    best_metrics = all_results[best_model_name]
    record = {
        "target": y_col,
        "skipped": False,
        "best_model": best_model_name,
        "n_splits": best_metrics.get("n_splits", float(n_splits_target)),
        "f1": best_metrics.get("f1", np.nan),
        "precision": best_metrics.get("precision", np.nan),
        "recall": best_metrics.get("recall", np.nan),
        "accuracy": best_metrics.get("accuracy", np.nan),
        "auc": best_metrics.get("auc", np.nan),
    }
    logger.info("Final record for %s: %s", y_col, record)
    return record


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    id_cols, ft_cols, y_cols = detect_columns(df)
    log.info(
        "Detected %d id_, %d ft_, %d y_ columns",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not y_cols:
        log.error("No y_ columns found. Exiting.")
        return

    log.info(
        "Starting Stage-2 training for %d targets with N_JOBS_TARGETS=%d",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(process_target)(y_col, df) for y_col in y_cols
    )

    # Split into processed vs skipped
    processed_records = [r for r in results if not r.get("skipped", False)]
    skipped_records = [r for r in results if r.get("skipped", False)]

    if processed_records:
        df_results = pd.DataFrame(processed_records)
        df_results.to_csv(RESULTS_CSV, index=False)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(processed_records, f, indent=2)
        log.info("Saved Stage-2 metrics to %s and %s", RESULTS_CSV, RESULTS_JSON)
    else:
        log.warning("No targets successfully trained in Stage-2.")

    if skipped_records:
        df_skipped = pd.DataFrame(skipped_records)
        df_skipped.to_csv(SKIPPED_CSV, index=False)
        log.info("Saved skipped targets list to %s", SKIPPED_CSV)


if __name__ == "__main__":
    main()
