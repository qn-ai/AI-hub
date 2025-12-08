#!/usr/bin/env python
"""
Stage-1: Per-Target Feature Importances with Rare-Class Skipping.

This script computes feature importances for each y_* target, but:
- Skips targets with fewer than MIN_SAMPLES_PER_TARGET labelled rows.
- Skips targets where the smallest class has < MIN_CLASS_COUNT_FOR_IMPORTANCE samples
  (e.g. only 1 sample, or only 1 class total).

For each valid target y_<name>:

1. Extract non-missing rows for that target.
2. Check class distribution:
   - if n_classes < 2 or min_class_count < MIN_CLASS_COUNT_FOR_IMPORTANCE,
     skip this target as "rare / degenerate".
3. Build two feature views:
   - Encoded numeric view (CatBoostEncoder) for RF, LGBM, XGB, HGB.
   - Raw categorical view (string + NaNs -> "NA_CAT") for CatBoost.
4. Train 5 NaN-aware models:
   - RandomForestClassifier (sklearn >= 1.4: https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_4_0.html)
   - LGBMClassifier
   - XGBClassifier
   - HistGradientBoostingClassifier
   - CatBoostClassifier
5. Extract importances and aggregate to a mean_rank per feature.
6. Save to: feature_importances/feature_importances_<y>.csv with columns:
   - feature_name, RF, LGBM, CB, XGB, HGB, mean_rank

Additionally:
- A file feature_importances/skipped_targets_stage1.csv records all skipped targets
  and reasons (too few rows, rare classes, etc.).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
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

# Global feature cleanup
USE_GLOBAL_VAR_CORR_CLEANUP = True
VAR_THRESH = 0.0
CORR_THRESH = 0.95

# Per-target missing-feature cleanup
MISSING_THRESH = 0.8  # drop ft_* with > 80% missing for that target

# Skipping thresholds for targets
MIN_SAMPLES_PER_TARGET = 200
MIN_CLASS_COUNT_FOR_IMPORTANCE = 2  # if smallest class < 2 → skip target

# Parallel config (models themselves may parallelise internally)
_CPU = os.cpu_count() or 4

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SKIPPED_CSV = FEATURE_IMPORTANCE_DIR / "skipped_targets_stage1.csv"

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage1")


def get_target_logger(y_col: str) -> logging.Logger:
    """Per-target logger → logs/y_<target>_stage1.log."""
    logger = logging.getLogger(f"stage1.{y_col}")
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage1_file", False)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(LOG_DIR / f"{y_col}_stage1.log", mode="w", encoding="utf-8")
        fh._stage1_file = True  # type: ignore[attr-defined]
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = True
    return logger


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def global_feature_cleanup(X: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove zero-variance and highly correlated numeric columns (global)."""
    if not USE_GLOBAL_VAR_CORR_CLEANUP:
        logger.info("Global cleanup disabled; keeping all %d ft_ columns.", X.shape[1])
        return X

    logger.info("Global cleanup: starting with %d features.", X.shape[1])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Zero variance
    if num_cols:
        vt = VarianceThreshold(threshold=VAR_THRESH)
        vt.fit(X[num_cols])
        keep_mask = vt.get_support()
        keep_cols = [c for c, k in zip(num_cols, keep_mask) if k]
        drop_cols = [c for c in num_cols if c not in keep_cols]
        if drop_cols:
            logger.info("Dropping %d zero-variance numeric features.", len(drop_cols))
            X = X.drop(columns=drop_cols)
            num_cols = keep_cols

    # High correlation
    if len(num_cols) > 1:
        corr = X[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESH)]
        if to_drop:
            logger.info(
                "Dropping %d highly correlated numeric features (>|%.2f|).",
                len(to_drop),
                CORR_THRESH,
            )
            X = X.drop(columns=to_drop)

    logger.info("Global cleanup done: %d features remain.", X.shape[1])
    return X


def per_target_missing_cleanup(X: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Drop columns with too many NaNs for this target subset."""
    missing_ratio = X.isna().mean()
    drop_cols = missing_ratio[missing_ratio > MISSING_THRESH].index.tolist()
    if drop_cols:
        logger.info(
            "Per-target cleanup: dropping %d columns with missing_ratio > %.2f",
            len(drop_cols),
            MISSING_THRESH,
        )
        X = X.drop(columns=drop_cols)
    return X


def prepare_target_for_importance(
    y_raw: pd.Series,
    logger: logging.Logger,
) -> Optional[pd.Series]:
    """Encode target to contiguous 0..K-1; skip degenerate / rare targets.

    Rules:
    - Drop NaNs (handled before).
    - Compute class counts.
    - If n_classes < 2 → skip (only one label).
    - If smallest class < MIN_CLASS_COUNT_FOR_IMPORTANCE → skip (too rare).
    - Otherwise, encode to ints 0..K-1 via LabelEncoder on strings.
    """
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    n_classes = counts.shape[0]
    min_count = int(counts.min())
    logger.info("Class distribution: %s", counts.to_dict())

    if n_classes < 2:
        logger.warning(
            "Skipping target: only one class present (n_classes=1)."
        )
        return None

    if min_count < MIN_CLASS_COUNT_FOR_IMPORTANCE:
        logger.warning(
            "Skipping target: min class count = %d < MIN_CLASS_COUNT_FOR_IMPORTANCE=%d.",
            min_count,
            MIN_CLASS_COUNT_FOR_IMPORTANCE,
        )
        return None

    le = LabelEncoder()
    y_enc = pd.Series(
        le.fit_transform(y_str),
        index=y_raw.index,
        dtype="int64",
    )
    logger.info(
        "Encoded labels: %s (K=%d, min_count=%d)",
        np.unique(y_enc).tolist(),
        n_classes,
        min_count,
    )
    return y_enc


def prepare_views_for_importance(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_num, X_cb) numeric + CatBoost views."""
    # Numeric view with CatBoostEncoder
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        dummy_y = np.zeros(len(X_num))
        X_num = enc.fit_transform(X_num, dummy_y)
    elif not USE_CATBOOST_ENCODER and cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    # CatBoost view: string categoricals
    X_cb = X.copy()
    cb_cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for c in cb_cat_cols:
        X_cb[c] = X_cb[c].astype("string").fillna(CAT_FILL_VALUE)

    return X_num, X_cb


def build_importance_models(is_binary: bool) -> Dict[str, object]:
    """Model prototypes for feature importance."""
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


def get_xgb_importance(model: XGBClassifier, feature_names: List[str]) -> pd.Series:
    """Map XGBoost booster feature scores to our feature names."""
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    imp = pd.Series(0.0, index=feature_names)

    for fname, score in raw.items():
        if fname in imp.index:
            imp[fname] = score
            continue
        if fname.startswith("f") and fname[1:].isdigit():
            idx = int(fname[1:])
            if idx < len(feature_names):
                imp[feature_names[idx]] = score

    if imp.sum() > 0:
        imp = imp / imp.sum()
    return imp


def compute_importances_for_target(
    X_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    y: pd.Series,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Train importance models and return DataFrame[feature_name, RF, LGBM, CB, XGB, HGB]."""
    is_binary = y.nunique() == 2
    feature_names = X_num.columns.tolist()
    models = build_importance_models(is_binary)

    # RF
    logger.info("Training RF for importances.")
    rf = models["RF"].__class__(**models["RF"].get_params())
    rf.fit(X_num, y)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names)

    # LGBM
    logger.info("Training LGBM for importances.")
    lgbm = models["LGBM"].__class__(**models["LGBM"].get_params())
    lgbm.fit(X_num, y)
    lgbm_imp = pd.Series(lgbm.feature_importances_, index=feature_names)

    # XGB
    logger.info("Training XGB for importances.")
    xgb = models["XGB"].__class__(**models["XGB"].get_params())
    xgb.fit(X_num, y)
    xgb_imp = get_xgb_importance(xgb, feature_names)

    # HGB
    logger.info("Training HGB for importances.")
    hgb = models["HGB"].__class__(**models["HGB"].get_params())
    hgb.fit(X_num, y)
    hgb_imp = pd.Series(hgb.feature_importances_, index=feature_names)

    # CatBoost
    logger.info("Training CatBoost for importances.")
    cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
    cat_indices = [X_cb.columns.get_loc(c) for c in cb_cat_cols]
    cb = models["CB"].__class__(**models["CB"].get_params())
    cb.fit(X_cb, y, cat_features=cat_indices if cat_indices else None)
    cb_imp_raw = cb.get_feature_importance()
    cb_imp = pd.Series(cb_imp_raw, index=X_cb.columns).reindex(feature_names).fillna(0.0)

    df_imp = pd.DataFrame(
        {
            "feature_name": feature_names,
            "RF": rf_imp.values,
            "LGBM": lgbm_imp.values,
            "CB": cb_imp.values,
            "XGB": xgb_imp.reindex(feature_names).fillna(0.0).values,
            "HGB": hgb_imp.values,
        }
    )
    return df_imp


# ---------------------------------------------------------------------
# MAIN PER-TARGET LOOP
# ---------------------------------------------------------------------

def main() -> None:
    FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    id_cols, ft_cols, y_cols = detect_columns(df)
    log.info(
        "Detected %d id_, %d ft_, %d y_ columns.",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not ft_cols or not y_cols:
        log.error("Missing ft_ or y_ columns; nothing to do.")
        return

    # Global feature cleanup once
    X_all = df[ft_cols].copy()
    X_base = global_feature_cleanup(X_all, logger=log)
    base_feature_names = X_base.columns.tolist()

    skipped_records: List[Dict[str, str]] = []

    for y_col in y_cols:
        logger = get_target_logger(y_col)
        logger.info("=== Stage-1 feature importance for %s ===", y_col)

        df_target = df[df[y_col].notna()].copy()
        n_rows = df_target.shape[0]
        if n_rows < MIN_SAMPLES_PER_TARGET:
            logger.warning(
                "Skipping %s: only %d labelled rows (< MIN_SAMPLES_PER_TARGET=%d).",
                y_col,
                n_rows,
                MIN_SAMPLES_PER_TARGET,
            )
            skipped_records.append(
                {
                    "target": y_col,
                    "reason": "too_few_rows",
                    "n_rows": str(n_rows),
                }
            )
            continue

        y_raw = df_target[y_col]
        y_enc = prepare_target_for_importance(y_raw, logger)
        if y_enc is None:
            skipped_records.append(
                {
                    "target": y_col,
                    "reason": "rare_or_single_class",
                    "n_rows": str(n_rows),
                }
            )
            continue

        # Align global-cleaned features with these rows
        X_t = X_base.loc[df_target.index].copy()
        if X_t.shape[1] == 0:
            logger.warning("Skipping %s: no features after global cleanup.", y_col)
            skipped_records.append(
                {
                    "target": y_col,
                    "reason": "no_features_after_cleanup",
                    "n_rows": str(n_rows),
                }
            )
            continue

        # Drop high-missing columns for this target
        X_t = per_target_missing_cleanup(X_t, logger)
        if X_t.shape[1] == 0:
            logger.warning("Skipping %s: all features dropped by missing filter.", y_col)
            skipped_records.append(
                {
                    "target": y_col,
                    "reason": "all_features_high_missing",
                    "n_rows": str(n_rows),
                }
            )
            continue

        # Prepare views
        X_num, X_cb = prepare_views_for_importance(X_t)
        if X_num.shape[1] == 0:
            logger.warning("Skipping %s: no numeric features after encoding.", y_col)
            skipped_records.append(
                {
                    "target": y_col,
                    "reason": "no_numeric_features",
                    "n_rows": str(n_rows),
                }
            )
            continue

        # Compute importances
        df_imp = compute_importances_for_target(
            X_num=X_num,
            X_cb=X_cb,
            y=y_enc,
            logger=logger,
        )

        # Rank & save
        rank_df = df_imp[["RF", "LGBM", "CB", "XGB", "HGB"]].rank(
            method="average", ascending=False
        )
        df_imp["mean_rank"] = rank_df.mean(axis=1)

        df_sorted = df_imp.sort_values("mean_rank", ascending=True)
        out_path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
        df_sorted.to_csv(out_path, index=False)
        logger.info("Saved feature importance for %s to %s", y_col, out_path)

    if skipped_records:
        df_skipped = pd.DataFrame(skipped_records)
        df_skipped.to_csv(SKIPPED_CSV, index=False)
        log.info("Saved skipped targets for Stage-1 to %s", SKIPPED_CSV)
    else:
        log.info("No targets were skipped in Stage-1.")


if __name__ == "__main__":
    main()
