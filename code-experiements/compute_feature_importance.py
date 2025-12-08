#!/usr/bin/env python
"""
Stage-1: Per-Target Feature Importances with Safe Handling of Rare Classes,
NaNs, Mixed Feature Types, and CatBoost-Compatible Categorical Treatment.

This script:
- Loops through all y_* targets.
- Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
- Skips targets with only one class, or smallest class < MIN_CLASS_COUNT_FOR_IMPORTANCE.
- Cleans features globally (variance + correlation) and per-target (missing thresholds).
- Builds two feature views per target:
    * Numeric view: CatBoostEncoder → numeric + NaN (RF/LGBM/XGB/HGB)
    * CatBoost raw view: string categoricals + NaN → "NA_CAT"
- Trains 5 NaN-aware models:
    RandomForest, LightGBM, XGBoost, HistGradientBoosting, CatBoost
- Extracts feature importances per model:
    * RF: feature_importances_
    * LGBM: feature_importances_
    * XGB: booster.get_score mapped to feature names
    * HGB: sklearn.inspection.permutation_importance
    * CatBoost: model.get_feature_importance()
- Saves:
    feature_importances/feature_importances_<y>.csv with columns:
        feature_name, RF, LGBM, CB, XGB, HGB, mean_rank
- Skipped targets saved to:
    feature_importances/skipped_targets_stage1.csv
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
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_PATH = "input_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
LOG_DIR = Path("logs")

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

RANDOM_STATE = 42

# Stage-1 global cleanup
USE_GLOBAL_VAR_CORR_CLEANUP = True
VAR_THRESH = 0.0
CORR_THRESH = 0.95

# Per-target missing cleanup (>80% missing dropped)
MISSING_THRESH = 0.8

# Target skipping thresholds
MIN_SAMPLES_PER_TARGET = 200
MIN_CLASS_COUNT_FOR_IMPORTANCE = 2   # smallest class must have ≥2 samples

CPU = os.cpu_count() or 4
USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

FEATURE_IMPORTANCE_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

SKIPPED_CSV = FEATURE_IMPORTANCE_DIR / "skipped_targets_stage1.csv"

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("stage1")

def get_target_logger(y_col: str) -> logging.Logger:
    """Create per-target logfile logs/y_<target>_stage1.log."""
    logger = logging.getLogger(f"stage1.{y_col}")
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage1_file", False)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(LOG_DIR / f"{y_col}_stage1.log", mode="w", encoding="utf-8")
        fh._stage1_file = True  # mark as ours
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.propagate = True
    return logger

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def detect_columns(df: pd.DataFrame):
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def global_feature_cleanup(X: pd.DataFrame, logger: logging.Logger):
    """VarianceThreshold + Correlation cleanup."""
    if not USE_GLOBAL_VAR_CORR_CLEANUP:
        logger.info("Global cleanup disabled.")
        return X

    logger.info("Global cleanup: initial features = %d", X.shape[1])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Zero-variance numeric
    if num_cols:
        vt = VarianceThreshold(VAR_THRESH)
        vt.fit(X[num_cols])
        keep = vt.get_support()
        keep_cols = [c for c, k in zip(num_cols, keep) if k]
        drop_cols = [c for c in num_cols if c not in keep_cols]
        if drop_cols:
            logger.info("Dropping %d zero-variance numeric features.", len(drop_cols))
            X = X.drop(columns=drop_cols)
            num_cols = keep_cols

    # High correlation removal
    if len(num_cols) > 1:
        corr = X[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
        if drop_cols:
            logger.info(
                "Dropping %d numeric features with |corr| > %.2f.",
                len(drop_cols),
                CORR_THRESH,
            )
            X = X.drop(columns=drop_cols)

    logger.info("Global cleanup: final features = %d", X.shape[1])
    return X


def per_target_missing_cleanup(X: pd.DataFrame, logger: logging.Logger):
    """Drop columns with >80% missing values in this target's subset."""
    missing_ratio = X.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MISSING_THRESH].index.tolist()
    if drop_cols:
        logger.info(
            "Dropping %d features with missing ratio > %.2f",
            len(drop_cols),
            MISSING_THRESH,
        )
        X = X.drop(columns=drop_cols)
    return X


def prepare_target_for_importance(y_raw: pd.Series, logger: logging.Logger) -> Optional[pd.Series]:
    """
    Skip target if:
      - only one class OR
      - smallest class < MIN_CLASS_COUNT_FOR_IMPORTANCE
    Otherwise label-encode to 0..K-1.
    """
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    min_count = int(counts.min())
    n_classes = counts.shape[0]

    logger.info("Class distribution: %s", counts.to_dict())

    if n_classes < 2:
        logger.warning("Skipping target: only one class present.")
        return None

    if min_count < MIN_CLASS_COUNT_FOR_IMPORTANCE:
        logger.warning(
            "Skipping target: smallest class has only %d samples (< %d).",
            min_count,
            MIN_CLASS_COUNT_FOR_IMPORTANCE,
        )
        return None

    le = LabelEncoder()
    y_enc = pd.Series(
        le.fit_transform(y_str),
        index=y_raw.index,
        dtype="int64"
    )
    logger.info("Encoded labels: %s", np.unique(y_enc).tolist())
    return y_enc


def prepare_views(X: pd.DataFrame):
    """Return numeric_view, catboost_view"""
    # Numeric view
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        dummy_y = np.zeros(len(X_num))
        X_num = enc.fit_transform(X_num, dummy_y)
    elif cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    # CatBoost raw view
    X_cb = X.copy()
    cb_cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for c in cb_cat_cols:
        X_cb[c] = X_cb[c].astype("string").fillna(CAT_FILL_VALUE)

    return X_num, X_cb


def build_importance_models(is_binary: bool):
    """Return dict of model prototypes."""
    if is_binary:
        lgb_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgb_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    return {
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            objective=lgb_obj,
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


def map_xgb_importances(model: XGBClassifier, feature_names: List[str]) -> pd.Series:
    """Map booster.get_score(importance_type='gain') to correct feature names."""
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    out = pd.Series(0.0, index=feature_names)

    for fname, score in raw.items():
        if fname in out.index:
            out[fname] = score
        elif fname.startswith("f") and fname[1:].isdigit():
            idx = int(fname[1:])
            if idx < len(feature_names):
                out[feature_names[idx]] = score

    if out.sum() > 0:
        out = out / out.sum()
    return out


def compute_importances_for_target(
    X_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    y: pd.Series,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Train 5 models and produce feature importance dataframe."""
    is_binary = (y.nunique() == 2)
    feature_names = X_num.columns.tolist()
    models = build_importance_models(is_binary)

    # RF
    logger.info("Training RF…")
    rf = models["RF"].__class__(**models["RF"].get_params())
    rf.fit(X_num, y)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names)

    # LGBM
    logger.info("Training LGBM…")
    lgbm = models["LGBM"].__class__(**models["LGBM"].get_params())
    lgbm.fit(X_num, y)
    lgbm_imp = pd.Series(lgbm.feature_importances_, index=feature_names)

    # XGB
    logger.info("Training XGB…")
    xgb = models["XGB"].__class__(**models["XGB"].get_params())
    xgb.fit(X_num, y)
    xgb_imp = map_xgb_importances(xgb, feature_names)

    # HGB - no feature_importances_, use permutation_importance
    logger.info("Training HGB + permutation importance…")
    hgb = models["HGB"].__class__(**models["HGB"].get_params())
    hgb.fit(X_num, y)

    hgb_perm = permutation_importance(
        hgb,
        X_num,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    hgb_imp = pd.Series(hgb_perm.importances_mean, index=feature_names)

    # CatBoost
    logger.info("Training CatBoost…")
    cb = models["CB"].__class__(**models["CB"].get_params())
    cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
    cat_indices = [X_cb.columns.get_loc(c) for c in cb_cat_cols]
    cb.fit(X_cb, y, cat_features=cat_indices if cat_indices else None)

    cb_raw = cb.get_feature_importance()
    cb_imp = pd.Series(cb_raw, index=X_cb.columns).reindex(feature_names).fillna(0.0)

    df_imp = pd.DataFrame({
        "feature_name": feature_names,
        "RF": rf_imp.values,
        "LGBM": lgbm_imp.values,
        "CB": cb_imp.values,
        "XGB": xgb_imp.values,
        "HGB": hgb_imp.values,
    })

    return df_imp

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    log.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    id_cols, ft_cols, y_cols = detect_columns(df)
    log.info("Detected %d ft_ features and %d y_ targets.", len(ft_cols), len(y_cols))

    X_all = df[ft_cols].copy()
    X_base = global_feature_cleanup(X_all, log)
    base_feature_names = X_base.columns.tolist()

    skipped = []

    for y_col in y_cols:
        logger = get_target_logger(y_col)
        logger.info("=== Stage-1 for %s ===", y_col)

        df_t = df[df[y_col].notna()].copy()
        if df_t.shape[0] < MIN_SAMPLES_PER_TARGET:
            logger.warning(
                "Skipping %s: only %d labelled rows (< %d).",
                y_col, df_t.shape[0], MIN_SAMPLES_PER_TARGET
            )
            skipped.append({
                "target": y_col,
                "reason": "too_few_rows",
                "n_rows": df_t.shape[0],
            })
            continue

        y_raw = df_t[y_col]
        y_enc = prepare_target_for_importance(y_raw, logger)
        if y_enc is None:
            skipped.append({
                "target": y_col,
                "reason": "rare_or_single_class",
                "n_rows": df_t.shape[0],
            })
            continue

        # Align features
        X_t = X_base.loc[df_t.index].copy()
        X_t = per_target_missing_cleanup(X_t, logger)

        if X_t.shape[1] == 0:
            logger.warning("Skipping %s: no features remain after cleanup.", y_col)
            skipped.append({
                "target": y_col,
                "reason": "no_features_after_cleanup",
                "n_rows": df_t.shape[0],
            })
            continue

        # Prepare views
        X_num, X_cb = prepare_views(X_t)

        # Compute importances
        df_imp = compute_importances_for_target(X_num, X_cb, y_enc, logger)

        # Compute mean rank
        ranks = df_imp[["RF", "LGBM", "CB", "XGB", "HGB"]].rank(
            method="average",
            ascending=False
        )
        df_imp["mean_rank"] = ranks.mean(axis=1)

        # Save
        out_path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
        df_imp.sort_values("mean_rank", ascending=True).to_csv(out_path, index=False)
        logger.info("Saved importances to %s", out_path)

    # Save skipped summary
    if skipped:
        pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)
        log.info("Saved skipped targets to: %s", SKIPPED_CSV)
    else:
        log.info("No targets skipped.")

if __name__ == "__main__":
    main()
