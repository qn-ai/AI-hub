#!/usr/bin/env python3
"""
Stage-1: Per-target feature importances with safe handling of rare classes,
NaNs, mixed feature types, and CatBoost-compatible categorical treatment.

This script:

- Loops through all y_* targets.
- Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
- Skips targets with only one class, or smallest class
  < MIN_CLASS_COUNT_FOR_IMPORTANCE.
- Cleans features globally (variance + correlation) and
  per-target (missing thresholds).
- Builds two feature views per target:
    * Numeric view:
        - CatBoostEncoder → numeric + NaN (RF/LGBM/XGB/HGB/RF_REG).
    * CatBoost raw view:
        - string categoricals + NaN → "NA_CAT" (for CatBoost).
- Trains NaN-aware models (configurable):
    * RandomForestClassifier        ("RF")
    * RandomForestRegressor         ("RF_REG")
    * LGBMClassifier                ("LGBM")
    * XGBClassifier                 ("XGB")
    * HistGradientBoostingClassifier ("HGB")
    * CatBoostClassifier            ("CB")
- Extracts feature importances per model:
    * RF / RF_REG: feature_importances_
    * LGBM: feature_importances_
    * XGB: booster.get_score mapped to feature names
    * HGB: sklearn.inspection.permutation_importance
    * CatBoost: model.get_feature_importance()
- Saves:
    feature_importances/feature_importances_<y>.csv with columns:
        feature_name, <one column per enabled model>, mean_rank
- Skipped targets saved to:
    feature_importances/skipped_targets_stage1.csv

Optional:
- MLflow tracking per target (off by default).
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
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Optional MLflow
try:
    import mlflow  # type: ignore[import]

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - MLflow optional
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_PATH = "input_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
LOG_DIR = Path("logs")

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

RANDOM_STATE = 42

# Global feature cleanup
USE_GLOBAL_CLEANUP = True
VAR_THRESH = 0.0
CORR_THRESH = 0.95

# Per-target missing filter (>80% NaNs dropped)
MISSING_THRESH = 0.8

# Target skipping
MIN_SAMPLES_PER_TARGET = 200
MIN_CLASS_COUNT_FOR_IMPORTANCE = 2

CPU_COUNT = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

# Which importance models to run in Stage-1
# Valid entries: "RF", "RF_REG", "LGBM", "XGB", "HGB", "CB"
ENABLED_MODELS: List[str] = ["RF", "LGBM", "XGB", "HGB", "CB", "RF_REG"]

# MLflow toggle
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "Stage1_FeatureImportances"

FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SKIPPED_TARGETS_CSV = FEATURE_IMPORTANCE_DIR / "skipped_targets_stage1.csv"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("stage1")


def get_target_logger(y_col: str) -> logging.Logger:
    """Create a per-target logger writing to logs/y_<target>_stage1.log."""
    logger = logging.getLogger(f"stage1.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage1_file", False)
        for h in logger.handlers
    )
    if not exists:
        file_handler = logging.FileHandler(
            LOG_DIR / f"{y_col}_stage1.log",
            mode="w",
            encoding="utf-8",
        )
        file_handler._stage1_file = True  # type: ignore[attr-defined]
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


def global_feature_cleanup(
    X: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Perform global variance + correlation cleanup on numeric ft_ columns."""
    if not USE_GLOBAL_CLEANUP:
        logger.info("Global cleanup disabled; keeping all %d features.", X.shape[1])
        return X

    logger.info("Global cleanup: initial feature count = %d", X.shape[1])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Zero-variance removal
    if numeric_cols:
        vt = VarianceThreshold(threshold=VAR_THRESH)
        vt.fit(X[numeric_cols])
        keep_mask = vt.get_support()
        keep_cols = [c for c, keep in zip(numeric_cols, keep_mask) if keep]
        drop_cols = [c for c in numeric_cols if c not in keep_cols]
        if drop_cols:
            logger.info(
                "Dropping %d zero-variance numeric features.",
                len(drop_cols),
            )
            X = X.drop(columns=drop_cols)
            numeric_cols = keep_cols

    # High correlation removal
    if len(numeric_cols) > 1:
        corr = X[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
        if to_drop:
            logger.info(
                "Dropping %d numeric features with |corr| > %.2f.",
                len(to_drop),
                CORR_THRESH,
            )
            X = X.drop(columns=to_drop)

    logger.info("Global cleanup: final feature count = %d", X.shape[1])
    return X


def per_target_missing_cleanup(
    X: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Drop features with too many NaNs for this target subset."""
    missing_ratio = X.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MISSING_THRESH].index.tolist()
    if drop_cols:
        logger.info(
            "Per-target cleanup: dropping %d features (missing_ratio > %.2f).",
            len(drop_cols),
            MISSING_THRESH,
        )
        X = X.drop(columns=drop_cols)
    return X


def encode_target_for_importance(
    y_raw: pd.Series,
    logger: logging.Logger,
) -> Tuple[Optional[pd.Series], int, int]:
    """Encode target to integers and decide if it is usable.

    Returns:
        Tuple (y_encoded, n_classes, min_class_count).
        If y_encoded is None, the target should be skipped.
    """
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    n_classes = counts.shape[0]
    min_count = int(counts.min())

    logger.info("Target class distribution: %s", counts.to_dict())

    if n_classes < 2:
        logger.warning("Skipping: only one class present (n_classes=1).")
        return None, n_classes, min_count

    if min_count < MIN_CLASS_COUNT_FOR_IMPORTANCE:
        logger.warning(
            "Skipping: smallest class has only %d samples (< %d).",
            min_count,
            MIN_CLASS_COUNT_FOR_IMPORTANCE,
        )
        return None, n_classes, min_count

    encoder = LabelEncoder()
    y_encoded = pd.Series(
        encoder.fit_transform(y_str),
        index=y_raw.index,
        dtype="int64",
    )
    logger.info(
        "Label-encoded target: n_classes=%d, min_class_count=%d, labels=%s",
        n_classes,
        min_count,
        np.unique(y_encoded).tolist(),
    )
    return y_encoded, n_classes, min_count


def prepare_feature_views(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare numeric and CatBoost feature views.

    Returns:
        Tuple (X_num, X_cb).
    """
    # Numeric view
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        dummy_y = np.zeros(len(X_num))
        X_num = encoder.fit_transform(X_num, dummy_y)
    elif cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    # CatBoost view
    X_cb = X.copy()
    cb_cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        X_cb[col] = X_cb[col].astype("string").fillna(CAT_FILL_VALUE)

    return X_num, X_cb


def build_importance_models(is_binary: bool) -> Dict[str, object]:
    """Build lighter, NaN-aware models used for Stage-1 importances.

    Which models are actually instantiated is controlled by ENABLED_MODELS.
    """
    models: Dict[str, object] = {}

    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    if "RF" in ENABLED_MODELS:
        models["RF"] = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if "RF_REG" in ENABLED_MODELS:
        models["RF_REG"] = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if "LGBM" in ENABLED_MODELS:
        models["LGBM"] = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.1,
            objective=lgbm_obj,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            num_leaves=31,
            min_data_in_leaf=20,
            min_gain_to_split=1e-3,
            max_bin=63,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if "XGB" in ENABLED_MODELS:
        models["XGB"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=xgb_obj,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if "HGB" in ENABLED_MODELS:
        models["HGB"] = HistGradientBoostingClassifier(
            max_depth=None,
            random_state=RANDOM_STATE,
        )

    if "CB" in ENABLED_MODELS:
        models["CB"] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function=cb_loss,
            random_state=RANDOM_STATE,
            verbose=False,
        )

    return models


def get_xgb_importance(
    model: XGBClassifier,
    feature_names: List[str],
) -> pd.Series:
    """Map XGBoost importance scores to actual feature names."""
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    importance = pd.Series(0.0, index=feature_names)

    for fname, score in raw.items():
        if fname in importance.index:
            importance[fname] = score
            continue
        if fname.startswith("f") and fname[1:].isdigit():
            idx = int(fname[1:])
            if 0 <= idx < len(feature_names):
                importance[feature_names[idx]] = score

    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance


def compute_importances_for_target(
    X_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    y: pd.Series,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Train enabled models and compute feature importances for one target."""
    is_binary = y.nunique() == 2
    feature_names = X_num.columns.tolist()
    models = build_importance_models(is_binary=is_binary)

    importance_dict: Dict[str, pd.Series] = {}

    for name, proto in models.items():
        logger.info("Training model %s for feature importances.", name)
        if name == "CB":
            cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
            cat_indices = [X_cb.columns.get_loc(c) for c in cb_cat_cols]
            model = proto.__class__(**proto.get_params())
            model.fit(
                X_cb,
                y,
                cat_features=cat_indices if cat_indices else None,
            )
            cb_raw = model.get_feature_importance()
            series = (
                pd.Series(cb_raw, index=X_cb.columns)
                .reindex(feature_names)
                .fillna(0.0)
            )
            importance_dict["CB"] = series
        elif name == "LGBM":
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)
            series = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict["LGBM"] = series
        elif name == "RF":
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)
            series = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict["RF"] = series
        elif name == "RF_REG":
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)
            series = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict["RF_REG"] = series
        elif name == "XGB":
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)
            series = get_xgb_importance(model, feature_names)
            importance_dict["XGB"] = series.reindex(feature_names).fillna(0.0)
        elif name == "HGB":
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)
            perm = permutation_importance(
                model,
                X_num,
                y,
                n_repeats=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            series = pd.Series(perm.importances_mean, index=feature_names)
            importance_dict["HGB"] = series

    df_imp = pd.DataFrame({"feature_name": feature_names})
    for name, series in importance_dict.items():
        df_imp[name] = series.values

    rank_cols = [c for c in df_imp.columns if c != "feature_name"]
    ranks = df_imp[rank_cols].rank(method="average", ascending=False)
    df_imp["mean_rank"] = ranks.mean(axis=1)
    df_sorted = df_imp.sort_values("mean_rank", ascending=True)
    return df_sorted


# ---------------------------------------------------------------------------
# PER-TARGET PIPELINE
# ---------------------------------------------------------------------------

def process_target(
    y_col: str,
    df: pd.DataFrame,
    X_base: pd.DataFrame,
) -> Dict[str, object]:
    """Process a single target column for Stage-1."""
    logger = get_target_logger(y_col)
    logger.info("=== Stage-1 feature importances for %s ===", y_col)

    df_t = df[df[y_col].notna()].copy()
    n_rows = df_t.shape[0]
    if n_rows < MIN_SAMPLES_PER_TARGET:
        logger.warning(
            "Skipping %s: only %d labelled rows (< %d).",
            y_col,
            n_rows,
            MIN_SAMPLES_PER_TARGET,
        )
        return {
            "target": y_col,
            "skipped": True,
            "reason": "too_few_rows",
            "n_rows": int(n_rows),
        }

    y_raw = df_t[y_col]
    y_enc, n_classes, min_class = encode_target_for_importance(y_raw, logger)
    if y_enc is None:
        return {
            "target": y_col,
            "skipped": True,
            "reason": "rare_or_single_class",
            "n_rows": int(n_rows),
            "n_classes": int(n_classes),
            "min_class_count": int(min_class),
        }

    X_t = X_base.loc[df_t.index].copy()
    logger.info("Features before per-target missing cleanup: %d", X_t.shape[1])
    X_t = per_target_missing_cleanup(X_t, logger)
    if X_t.shape[1] == 0:
        logger.warning("Skipping %s: all features dropped by missing filter.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "all_features_high_missing",
            "n_rows": int(n_rows),
        }

    X_num, X_cb = prepare_feature_views(X_t)
    if X_num.shape[1] == 0:
        logger.warning("Skipping %s: no numeric features after encoding.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "no_numeric_features",
            "n_rows": int(n_rows),
        }

    run = None
    if USE_MLFLOW and MLFLOW_AVAILABLE and mlflow is not None:
        run = mlflow.start_run(run_name=f"stage1_{y_col}", nested=False)
        mlflow.log_param("target", y_col)
        mlflow.log_param("n_rows", int(n_rows))
        mlflow.log_param("n_features_after_global", int(X_base.shape[1]))
        mlflow.log_param("n_features_after_missing", int(X_t.shape[1]))
        mlflow.log_param("n_classes", int(n_classes))
        mlflow.log_param("min_class_count", int(min_class))
        mlflow.log_param("enabled_models", ",".join(ENABLED_MODELS))

    df_sorted = compute_importances_for_target(X_num, X_cb, y_enc, logger)

    out_path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
    df_sorted.to_csv(out_path, index=False)
    logger.info("Saved feature importances for %s to %s", y_col, out_path)

    if USE_MLFLOW and MLFLOW_AVAILABLE and run is not None and mlflow is not None:
        mlflow.log_param("n_features_final", int(df_sorted.shape[0]))
        mlflow.log_artifact(str(out_path))
        mlflow.end_run()

    return {
        "target": y_col,
        "skipped": False,
        "reason": "",
        "n_rows": int(n_rows),
        "n_classes": int(n_classes),
        "min_class_count": int(min_class),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Stage-1 feature importance computation over all y_* targets."""
    LOG.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    id_cols, ft_cols, y_cols = detect_columns(df)
    LOG.info(
        "Detected %d id_, %d ft_, %d y_ columns.",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not ft_cols or not y_cols:
        LOG.error("No ft_ or y_ columns detected; aborting.")
        return

    X_all = df[ft_cols].copy()
    X_base = global_feature_cleanup(X_all, LOG)

    if USE_MLFLOW and MLFLOW_AVAILABLE and mlflow is not None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    LOG.info(
        "Starting Stage-1 over %d targets with N_JOBS_TARGETS=%d.",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(process_target)(y_col, df, X_base) for y_col in y_cols
    )

    skipped_records = [r for r in results if r.get("skipped")]
    if skipped_records:
        df_skipped = pd.DataFrame(skipped_records)
        df_skipped.to_csv(SKIPPED_TARGETS_CSV, index=False)
        LOG.info("Saved skipped targets summary to %s", SKIPPED_TARGETS_CSV)
    else:
        LOG.info("No targets skipped in Stage-1.")

    processed_count = sum(1 for r in results if not r.get("skipped"))
    LOG.info(
        "Stage-1 completed: %d processed, %d skipped.",
        processed_count,
        len(skipped_records),
    )


if __name__ == "__main__":
    main()
