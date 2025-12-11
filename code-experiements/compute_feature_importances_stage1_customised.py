#!/usr/bin/env python3
"""
Stage-1: Per-target feature importances with safe handling of rare classes,
NaNs, mixed feature types, and CatBoost-compatible categorical treatment.

This script:

- Loops through all y_* targets.
- In classification mode:
    * Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
    * Skips targets with only one class, or smallest class
      < MIN_CLASS_COUNT_FOR_IMPORTANCE.
- In regression mode:
    * Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
    * Skips non-numeric targets.
- Cleans features globally (variance + correlation) and
  per-target (missing thresholds).
- Builds two feature views per target:
    * Numeric view:
        - CatBoostEncoder → numeric + NaN
          (rf/lgbm/xgb/hgb for classification, rf_reg for regression).
    * CatBoost raw view:
        - string categoricals + NaN → "NA_CAT" (for CatBoost in classification).
- Trains NaN-aware models (configurable via enabled_models), identified by
  lowercase keys:
    Classification targets (TASK_MODE = "classification"):
        * "rf"   → RandomForestClassifier
        * "lgbm" → LGBMClassifier
        * "xgb"  → XGBClassifier
        * "hgb"  → HistGradientBoostingClassifier
        * "cb"   → CatBoostClassifier
    Regression targets (TASK_MODE = "regression"):
        * "rf_reg" → RandomForestRegressor
- Extracts feature importances per model:
    Classification:
        * rf: feature_importances_
        * lgbm: feature_importances_
        * xgb: booster.get_score mapped to feature names
        * hgb: sklearn.inspection.permutation_importance
        * cb: model.get_feature_importance()
    Regression:
        * rf_reg: feature_importances_
- Saves:
    feature_importances/feature_importances_<y>.csv with columns:
        feature_name, <one column per enabled model>, mean_rank
      where column names are upper-case aliases:
        RF, LGBM, XGB, HGB, CB for classification targets
        RF_REG (plus mean_rank) for regression targets.
- Skipped targets saved to:
    feature_importances/skipped_targets_stage1.csv

Optional:
- MLflow tracking per target (off by default).

Usage:
    # 1) Choose global task mode:
    #    - "classification": all y_* treated as classification
    #    - "regression": all y_* treated as regression (numeric only)
    TASK_MODE = "classification"

    # 2) Choose which importance models to run (by lowercase keys):
    #    Classification mode example:
    enabled_models = ["rf", "lgbm", "xgb", "hgb", "cb"]

    #    Regression mode example:
    #    (only rf_reg is meaningful here)
    enabled_models = ["rf_reg"]

    # 3) Run Stage-1:
    #    python compute_feature_importances_stage1.py
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

    mlflow_available = True
except Exception:  # pragma: no cover - MLflow optional
    mlflow = None  # type: ignore[assignment]
    mlflow_available = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Global task mode:
#   "classification" → all targets treated as classification
#   "regression"     → all targets treated as regression (numeric only)
TASK_MODE = "classification"  # or "regression"

data_path = "input_data.csv"
feature_importance_dir = Path("feature_importances")
log_dir = Path("logs")

id_prefix = "id_"
feature_prefix = "ft_"
target_prefix = "y_"

random_state = 42

# Global feature cleanup
use_global_cleanup = True
var_thresh = 0.0
corr_thresh = 0.95

# Per-target missing filter (>80% NaNs dropped)
missing_thresh = 0.8

# Target skipping
min_samples_per_target = 200
min_class_count_for_importance = 2  # only used in classification mode

cpu_count = os.cpu_count() or 4
n_jobs_targets = max(min(cpu_count - 1, 16), 2)

use_catboost_encoder = True
cat_fill_value = "NA_CAT"

# Which importance models to run in Stage-1
# Valid entries: "rf", "rf_reg", "lgbm", "xgb", "hgb", "cb"
# NOTE:
#   - classification mode: "rf", "lgbm", "xgb", "hgb", "cb" are used
#   - regression mode: only "rf_reg" is used (others ignored)
enabled_models: List[str] = ["rf", "lgbm", "xgb", "hgb", "cb", "rf_reg"]

# Mapping from internal keys to column names (keeps Stage-2 compatible)
model_col_names: Dict[str, str] = {
    "rf": "RF",
    "rf_reg": "RF_REG",
    "lgbm": "LGBM",
    "xgb": "XGB",
    "hgb": "HGB",
    "cb": "CB",
}

# MLflow toggle
use_mlflow = False
mlflow_experiment_name = "stage1_feature_importances"

feature_importance_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

skipped_targets_csv = feature_importance_dir / "skipped_targets_stage1.csv"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage1")


def get_target_logger(y_col: str) -> logging.Logger:
    """Create a per-target logger writing to logs/y_<target>_stage1.log."""
    logger = logging.getLogger(f"stage1.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "_stage1_file", False)
        for handler in logger.handlers
    )
    if not exists:
        file_handler = logging.FileHandler(
            log_dir / f"{y_col}_stage1.log",
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
    id_cols = [c for c in df.columns if c.startswith(id_prefix)]
    ft_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    y_cols = [c for c in df.columns if c.startswith(target_prefix)]
    return id_cols, ft_cols, y_cols


def global_feature_cleanup(
    features: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Perform global variance + correlation cleanup on numeric ft_ columns."""
    if not use_global_cleanup:
        logger.info(
            "Global cleanup disabled; keeping all %d features.",
            features.shape[1],
        )
        return features

    logger.info("Global cleanup: initial feature count = %d", features.shape[1])
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    # Zero-variance removal
    if numeric_cols:
        vt = VarianceThreshold(threshold=var_thresh)
        vt.fit(features[numeric_cols])
        keep_mask = vt.get_support()
        keep_cols = [c for c, keep in zip(numeric_cols, keep_mask) if keep]
        drop_cols = [c for c in numeric_cols if c not in keep_cols]
        if drop_cols:
            logger.info(
                "Dropping %d zero-variance numeric features.",
                len(drop_cols),
            )
            features = features.drop(columns=drop_cols)
            numeric_cols = keep_cols

    # High correlation removal
    if len(numeric_cols) > 1:
        corr = features[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
        if to_drop:
            logger.info(
                "Dropping %d numeric features with |corr| > %.2f.",
                len(to_drop),
                corr_thresh,
            )
            features = features.drop(columns=to_drop)

    logger.info("Global cleanup: final feature count = %d", features.shape[1])
    return features


def per_target_missing_cleanup(
    features: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Drop features with too many NaNs for this target subset."""
    missing_ratio = features.isna().mean()
    drop_cols = missing_ratio[missing_ratio > missing_thresh].index.tolist()
    if drop_cols:
        logger.info(
            "Per-target cleanup: dropping %d features (missing_ratio > %.2f).",
            len(drop_cols),
            missing_thresh,
        )
        features = features.drop(columns=drop_cols)
    return features


def encode_target_classification(
    y_raw: pd.Series,
    logger: logging.Logger,
) -> Tuple[Optional[pd.Series], int, int]:
    """Encode classification target to integers and decide if it is usable."""
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    n_classes = counts.shape[0]
    min_count = int(counts.min())

    logger.info("Target class distribution: %s", counts.to_dict())

    if n_classes < 2:
        logger.warning("Skipping: only one class present (n_classes=1).")
        return None, n_classes, min_count

    if min_count < min_class_count_for_importance:
        logger.warning(
            "Skipping: smallest class has only %d samples (< %d).",
            min_count,
            min_class_count_for_importance,
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
    features: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare numeric and CatBoost feature views for one target.

    Uses the true target y (classification labels or regression values)
    for CatBoostEncoder so encoded categoricals carry real signal.
    """
    # Numeric view (for rf/lgbm/xgb/hgb/rf_reg)
    numeric = features.copy()
    cat_cols = numeric.select_dtypes(include=["object"]).columns.tolist()

    if use_catboost_encoder and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
        numeric = encoder.fit_transform(numeric, y)
    elif cat_cols:
        numeric = numeric.drop(columns=cat_cols)

    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    # CatBoost view (for cb in classification mode only)
    cb_view = features.copy()
    cb_cat_cols = cb_view.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        cb_view[col] = cb_view[col].astype("string").fillna(cat_fill_value)

    return numeric, cb_view


def build_importance_models(task_mode: str, is_binary: bool) -> Dict[str, object]:
    """Build NaN-aware models used for Stage-1 importances."""
    models: Dict[str, object] = {}

    if task_mode == "regression":
        # Only rf_reg makes sense here (true numeric y)
        if "rf_reg" in enabled_models:
            models["rf_reg"] = RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )
        return models

    # Classification branch
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    if "rf" in enabled_models:
        models["rf"] = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )

    if "lgbm" in enabled_models:
        models["lgbm"] = LGBMClassifier(
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
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    if "xgb" in enabled_models:
        models["xgb"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=xgb_obj,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    if "hgb" in enabled_models:
        models["hgb"] = HistGradientBoostingClassifier(
            max_depth=None,
            random_state=random_state,
        )

    if "cb" in enabled_models:
        models["cb"] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function=cb_loss,
            random_state=random_state,
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
    numeric: pd.DataFrame,
    cb_view: pd.DataFrame,
    y: pd.Series,
    task_mode: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Train enabled models and compute feature importances for one target."""
    feature_names = numeric.columns.tolist()

    if task_mode == "regression":
        models = build_importance_models(task_mode="regression", is_binary=False)
        importance_dict: Dict[str, pd.Series] = {}

        for name, proto in models.items():
            logger.info("Training regression model %s for feature importances.", name)
            col_name = model_col_names[name]
            model = proto.__class__(**proto.get_params())
            model.fit(numeric, y)
            series_rf_reg = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict[col_name] = series_rf_reg

        df_imp = pd.DataFrame({"feature_name": feature_names})
        for col_name, series in importance_dict.items():
            df_imp[col_name] = series.values

        rank_cols = [c for c in df_imp.columns if c != "feature_name"]
        ranks = df_imp[rank_cols].rank(method="average", ascending=False)
        df_imp["mean_rank"] = ranks.mean(axis=1)
        df_sorted = df_imp.sort_values("mean_rank", ascending=True)
        return df_sorted

    # Classification branch
    is_binary = y.nunique() == 2
    models = build_importance_models(task_mode="classification", is_binary=is_binary)
    importance_dict: Dict[str, pd.Series] = {}

    for name, proto in models.items():
        logger.info("Training classification model %s for feature importances.", name)
        col_name = model_col_names[name]

        if name == "cb":
            cb_cat_cols = cb_view.select_dtypes(include=["string"]).columns.tolist()
            cat_indices = [cb_view.columns.get_loc(col) for col in cb_cat_cols]
            model = proto.__class__(**proto.get_params())
            model.fit(
                cb_view,
                y,
                cat_features=cat_indices if cat_indices else None,
            )
            raw_cb = model.get_feature_importance()
            series_cb = (
                pd.Series(raw_cb, index=cb_view.columns)
                .reindex(feature_names)
                .fillna(0.0)
            )
            importance_dict[col_name] = series_cb
        elif name == "lgbm":
            model = proto.__class__(**proto.get_params())
            model.fit(numeric, y)
            series_lgbm = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict[col_name] = series_lgbm
        elif name == "rf":
            model = proto.__class__(**proto.get_params())
            model.fit(numeric, y)
            series_rf = pd.Series(model.feature_importances_, index=feature_names)
            importance_dict[col_name] = series_rf
        elif name == "xgb":
            model = proto.__class__(**proto.get_params())
            model.fit(numeric, y)
            series_xgb = get_xgb_importance(model, feature_names)
            importance_dict[col_name] = series_xgb.reindex(feature_names).fillna(0.0)
        elif name == "hgb":
            model = proto.__class__(**proto.get_params())
            model.fit(numeric, y)
            perm = permutation_importance(
                model,
                numeric,
                y,
                n_repeats=3,
                random_state=random_state,
                n_jobs=-1,
            )
            series_hgb = pd.Series(perm.importances_mean, index=feature_names)
            importance_dict[col_name] = series_hgb

    df_imp = pd.DataFrame({"feature_name": feature_names})
    for col_name, series in importance_dict.items():
        df_imp[col_name] = series.values

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
    base_features: pd.DataFrame,
) -> Dict[str, object]:
    """Process a single target column for Stage-1."""
    logger = get_target_logger(y_col)
    logger.info("=== Stage-1 feature importances for %s ===", y_col)

    df_target = df[df[y_col].notna()].copy()
    n_rows = df_target.shape[0]
    if n_rows < min_samples_per_target:
        logger.warning(
            "Skipping %s: only %d labelled rows (< %d).",
            y_col,
            n_rows,
            min_samples_per_target,
        )
        return {
            "target": y_col,
            "skipped": True,
            "reason": "too_few_rows",
            "n_rows": int(n_rows),
        }

    y_raw = df_target[y_col]

    if TASK_MODE == "regression":
        if not pd.api.types.is_numeric_dtype(y_raw):
            logger.warning(
                "Skipping %s: non-numeric target in regression mode.",
                y_col,
            )
            return {
                "target": y_col,
                "skipped": True,
                "reason": "non_numeric_target_regression",
                "n_rows": int(n_rows),
            }
        y_reg = pd.to_numeric(y_raw, errors="coerce")
        if y_reg.isna().all():
            logger.warning(
                "Skipping %s: target becomes all-NaN after numeric coercion.",
                y_col,
            )
            return {
                "target": y_col,
                "skipped": True,
                "reason": "all_nan_after_coerce",
                "n_rows": int(n_rows),
            }
        y_valid_mask = y_reg.notna()
        df_target = df_target[y_valid_mask]
        y_reg = y_reg[y_valid_mask]
        n_rows = df_target.shape[0]
        logger.info(
            "Regression target %s: %d valid numeric rows after coercion.",
            y_col,
            n_rows,
        )
        if n_rows < min_samples_per_target:
            logger.warning(
                "Skipping %s: only %d valid rows (< %d) after coercion.",
                y_col,
                n_rows,
                min_samples_per_target,
            )
            return {
                "target": y_col,
                "skipped": True,
                "reason": "too_few_rows_after_coerce",
                "n_rows": int(n_rows),
            }
        y_for_encoding = y_reg
        n_classes = 0
        min_class = 0
    else:
        # classification mode
        y_class, n_classes, min_class = encode_target_classification(
            y_raw,
            logger,
        )
        if y_class is None:
            return {
                "target": y_col,
                "skipped": True,
                "reason": "rare_or_single_class",
                "n_rows": int(n_rows),
                "n_classes": int(n_classes),
                "min_class_count": int(min_class),
            }
        y_for_encoding = y_class

    features_t = base_features.loc[df_target.index].copy()
    logger.info(
        "Features before per-target missing cleanup: %d",
        features_t.shape[1],
    )
    features_t = per_target_missing_cleanup(features_t, logger)
    if features_t.shape[1] == 0:
        logger.warning("Skipping %s: all features dropped by missing filter.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "all_features_high_missing",
            "n_rows": int(n_rows),
        }

    numeric, cb_view = prepare_feature_views(features_t, y_for_encoding)
    if numeric.shape[1] == 0:
        logger.warning("Skipping %s: no numeric features after encoding.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "no_numeric_features",
            "n_rows": int(n_rows),
        }

    run = None
    if use_mlflow and mlflow_available and mlflow is not None:
        run = mlflow.start_run(run_name=f"stage1_{y_col}", nested=False)
        mlflow.log_param("target", y_col)
        mlflow.log_param("task_mode", TASK_MODE)
        mlflow.log_param("n_rows", int(n_rows))
        mlflow.log_param("n_features_after_global", int(base_features.shape[1]))
        mlflow.log_param("n_features_after_missing", int(features_t.shape[1]))
        if TASK_MODE == "classification":
            mlflow.log_param("n_classes", int(n_classes))
            mlflow.log_param("min_class_count", int(min_class))
        mlflow.log_param("enabled_models", ",".join(enabled_models))

    df_sorted = compute_importances_for_target(
        numeric=numeric,
        cb_view=cb_view,
        y=y_for_encoding,
        task_mode=TASK_MODE,
        logger=logger,
    )

    out_path = feature_importance_dir / f"feature_importances_{y_col}.csv"
    df_sorted.to_csv(out_path, index=False)
    logger.info("Saved feature importances for %s to %s", y_col, out_path)

    if use_mlflow and mlflow_available and run is not None and mlflow is not None:
        mlflow.log_param("n_features_final", int(df_sorted.shape[0]))
        mlflow.log_artifact(str(out_path))
        mlflow.end_run()

    return {
        "target": y_col,
        "skipped": False,
        "reason": "",
        "n_rows": int(n_rows),
        "task_mode": TASK_MODE,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Stage-1 feature importance computation over all y_* targets."""
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)

    id_cols, ft_cols, y_cols = detect_columns(df)
    log.info(
        "Detected %d id_, %d ft_, %d y_ columns.",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not ft_cols or not y_cols:
        log.error("No ft_ or y_ columns detected; aborting.")
        return

    all_features = df[ft_cols].copy()
    base_features = global_feature_cleanup(all_features, log)

    if use_mlflow and mlflow_available and mlflow is not None:
        mlflow.set_experiment(mlflow_experiment_name)

    log.info(
        "Starting Stage-1 over %d targets with n_jobs_targets=%d (task_mode=%s).",
        len(y_cols),
        n_jobs_targets,
        TASK_MODE,
    )

    results = Parallel(n_jobs=n_jobs_targets)(
        delayed(process_target)(y_col, df, base_features) for y_col in y_cols
    )

    skipped_records = [record for record in results if record.get("skipped")]
    if skipped_records:
        df_skipped = pd.DataFrame(skipped_records)
        df_skipped.to_csv(skipped_targets_csv, index=False)
        log.info("Saved skipped targets summary to %s", skipped_targets_csv)
    else:
        log.info("No targets skipped in Stage-1.")

    processed_count = sum(
        1 for record in results if not record.get("skipped")
    )
    log.info(
        "Stage-1 completed: %d processed, %d skipped.",
        processed_count,
        len(skipped_records),
    )


if __name__ == "__main__":
    main()
