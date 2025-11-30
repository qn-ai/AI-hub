#!/usr/bin/env python
"""Multi-target feature reduction using tree-based model importances.

This script assumes a wide table with:
- identifier columns starting with ``id_``,
- feature columns starting with ``ft_``,
- target columns starting with ``y_``.

Pipeline:
- Load the CSV dataset.
- Detect id_, ft_, y_ columns by prefix.
- Perform a single global feature cleanup on all ft_ columns:
    * drop high-missing columns,
    * drop zero-variance numeric columns,
    * drop highly-correlated numeric columns.
- For each target column (y_*) with enough non-missing rows:
    * subset rows where this y_ is not missing,
    * label-encode the target if it is non-numeric (for XGBoost etc.),
    * encode categoricals with CatBoostEncoder (no rare-category combining),
    * train RandomForest, LightGBM, CatBoost, and XGBoost,
    * compute feature importances for that target,
    * aggregate model importances into a per-target mean_rank,
    * save as ``feature_importances_<y_col>.csv``.
- Aggregate per-target mean_rank across all y_ columns into a global ranking
  and save as ``feature_importances_global_all_targets.csv``.

Usage:
    Adjust the CONFIG section, then run:

        python feature_reduction_multi_target.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =====================================================================
# CONFIG
# =====================================================================

DATA_PATH = "input_data.csv"  # Path to input CSV file.

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

MISSING_THRESH = 0.8  # Drop features with > 80% missing.
CORR_THRESH = 0.95  # Drop numeric features with |corr| > 0.95.
TOP_K_FEATURES = 100  # Number of top features to log as "top features".
MIN_SAMPLES_PER_TARGET = 200  # Skip targets with fewer labelled rows.

RANDOM_STATE = 42

OUTPUT_DIR = Path("feature_importances")  # Folder for CSV outputs.

# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# =====================================================================
# UTILS
# =====================================================================


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Detect identifier, feature, and target columns by prefix.

    Args:
        df: Input DataFrame.

    Returns:
        A tuple of three lists:
        - id_cols: Columns starting with ``ID_PREFIX``.
        - ft_cols: Columns starting with ``FEATURE_PREFIX``.
        - y_cols: Columns starting with ``TARGET_PREFIX``.
    """
    id_cols = [col for col in df.columns if col.startswith(ID_PREFIX)]
    ft_cols = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    y_cols = [col for col in df.columns if col.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def basic_feature_cleanup(
    X: pd.DataFrame,
    missing_thresh: float = MISSING_THRESH,
    corr_thresh: float = CORR_THRESH,
) -> pd.DataFrame:
    """Perform global basic feature reduction on ft_ columns.

    Operations:
      * Drop columns with too many missing values (global, across all rows).
      * Drop zero-variance numeric features.
      * Drop highly correlated numeric features (|corr| > corr_thresh).

    Args:
        X: Feature matrix with ft_ columns only.
        missing_thresh: Threshold for maximum allowed missing fraction.
        corr_thresh: Threshold for absolute correlation to drop features.

    Returns:
        A reduced feature matrix after cleanup (same rows, fewer columns).
    """
    logging.info("Initial feature count: %d", X.shape[1])

    # 1) Drop high-missing columns.
    missing_ratio = X.isna().mean()
    drop_missing = missing_ratio[missing_ratio > missing_thresh].index.tolist()
    if drop_missing:
        logging.info(
            "Dropping %d columns with missing_ratio > %.2f",
            len(drop_missing),
            missing_thresh,
        )
        X = X.drop(columns=drop_missing)

    # Split numeric vs categorical after missing-drop.
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 2) Drop zero-variance numeric features.
    if num_cols:
        vt = VarianceThreshold(threshold=0.0)
        X_num = X[num_cols]
        vt.fit(X_num)
        keep_mask = vt.get_support()
        keep_num_cols: List[str] = [
            col for col, keep in zip(num_cols, keep_mask) if keep
        ]
        drop_var = [col for col in num_cols if col not in keep_num_cols]
        if drop_var:
            logging.info(
                "Dropping %d numeric columns with zero variance", len(drop_var)
            )
            X = X.drop(columns=drop_var)
            num_cols = keep_num_cols

    # 3) Drop highly correlated numeric features.
    if len(num_cols) > 1:
        corr = X[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        if to_drop:
            logging.info(
                "Dropping %d numeric columns with corr > %.2f",
                len(to_drop),
                corr_thresh,
            )
            X = X.drop(columns=to_drop)

    logging.info("Feature count after cleanup: %d", X.shape[1])
    return X


def encode_categoricals_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: List[str],
) -> pd.DataFrame:
    """Encode categorical features using CatBoostEncoder.

    This does **not** combine rare categories. All category levels are
    preserved as-is and encoded into numeric features.

    Args:
        X: Feature matrix after basic cleanup (subset of rows).
        y: Target series aligned with ``X``.
        cat_cols: Names of categorical columns in ``X`` to encode.

    Returns:
        Encoded feature matrix (all numeric), with same column names as X.
    """
    if cat_cols:
        logging.info(
            "Encoding %d categorical columns with CatBoostEncoder for this "
            "target",
            len(cat_cols),
        )
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        X_enc = encoder.fit_transform(X, y)
        return X_enc

    logging.info("No categorical columns detected; skipping encoding")
    return X.copy()


def get_xgb_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Train XGBoost and compute feature importances (gain).

    Args:
        X: Encoded feature matrix (all numeric).
        y: Target series (numeric / label-encoded).

    Returns:
        A pandas Series of normalized feature importances indexed by column
        names of ``X``.
    """
    logging.info("Training XGBoost for feature importance")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    booster = model.get_booster()
    raw_score = booster.get_score(importance_type="gain")

    importance = pd.Series(0.0, index=X.columns)
    for fname, score in raw_score.items():
        # XGBoost uses feature names f0, f1, ...
        idx = int(fname[1:])
        if idx < len(X.columns):
            col_name = X.columns[idx]
            importance[col_name] = score

    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance


def compute_model_importances_for_target(
    X_enc: pd.DataFrame,
    y: pd.Series,
    original_X_for_cb: pd.DataFrame,
    cat_cols_cb: List[str],
) -> pd.DataFrame:
    """Compute feature importances from RF, LightGBM, CatBoost, and XGBoost.

    Args:
        X_enc:
            Encoded feature matrix (numeric) for RandomForest, LightGBM,
            and XGBoost.
        y:
            Target series for the current y_ column (numeric / label-encoded).
        original_X_for_cb:
            Original (non-encoded) feature matrix after basic cleanup
            for CatBoost.
        cat_cols_cb:
            Names of categorical columns in ``original_X_for_cb`` for
            CatBoost's native categorical handling.

    Returns:
        DataFrame with columns ``['RF', 'LGBM', 'CB', 'XGB']`` and features
        as the index, containing per-model importances.
    """
    feature_names = X_enc.columns

    # Random Forest
    logging.info("Training RandomForest for feature importance")
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_enc, y)
    imp_rf = pd.Series(rf.feature_importances_, index=feature_names)

    # LightGBM
    logging.info("Training LightGBM for feature importance")
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm.fit(X_enc, y)
    imp_lgb = pd.Series(lgbm.feature_importances_, index=feature_names)

    # XGBoost
    imp_xgb = get_xgb_importance(X_enc, y)

    # CatBoost (native categorical handling).
    logging.info("Training CatBoost for feature importance")
    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",  # For multiclass. Use 'Logloss' for binary.
        random_state=RANDOM_STATE,
        verbose=False,
    )

    cat_indices = (
        [original_X_for_cb.columns.get_loc(col) for col in cat_cols_cb]
        if cat_cols_cb
        else None
    )
    cb.fit(original_X_for_cb, y, cat_features=cat_indices)

    imp_cb_raw = cb.get_feature_importance()
    imp_cb = pd.Series(imp_cb_raw, index=original_X_for_cb.columns)

    # Align CatBoost importance to X_enc columns (subset).
    imp_cb_aligned = imp_cb.reindex(feature_names).fillna(0.0)

    df_imp = pd.DataFrame(
        {
            "RF": imp_rf,
            "LGBM": imp_lgb,
            "CB": imp_cb_aligned,
            "XGB": imp_xgb,
        }
    ).fillna(0.0)

    return df_imp


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    """Run the multi-target feature reduction pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    id_cols, ft_cols, y_cols = detect_columns(df)
    logging.info(
        "Detected %d id_, %d ft_, %d y_ columns",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not ft_cols:
        msg = "No feature columns (ft_*) detected."
        raise ValueError(msg)
    if not y_cols:
        msg = "No target columns (y_*) detected."
        raise ValueError(msg)

    # Global feature cleanup on all rows (unsupervised, same for all targets).
    X_all = df[ft_cols].copy()
    X_clean = basic_feature_cleanup(X_all)

    # Identify global numeric vs categorical columns after cleanup.
    num_cols_global = X_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_global = [col for col in X_clean.columns if col not in num_cols_global]
    logging.info(
        "After cleanup: %d numeric + %d categorical feature columns",
        len(num_cols_global),
        len(cat_cols_global),
    )

    feature_names = X_clean.columns

    # Global rank aggregation across targets.
    global_rank_sum = pd.Series(0.0, index=feature_names)
    global_rank_count = 0

    # Loop over all y_ targets.
    for y_col in y_cols:
        # Select rows where this target is not missing.
        df_target = df[df[y_col].notna()].copy()
        n_rows = df_target.shape[0]

        if n_rows < MIN_SAMPLES_PER_TARGET:
            logging.info(
                "Skipping %s: only %d non-missing rows (< %d)",
                y_col,
                n_rows,
                MIN_SAMPLES_PER_TARGET,
            )
            continue

        logging.info("Processing target: %s (rows with label: %d)", y_col, n_rows)

        y_raw = df_target[y_col]

        # Label-encode y if it is non-numeric (e.g., string classes).
        if not np.issubdtype(y_raw.dtype, np.number):
            logging.info("Label-encoding non-numeric target for %s", y_col)
            le = LabelEncoder()
            y = pd.Series(
                le.fit_transform(y_raw.astype(str)),
                index=y_raw.index,
            )
        else:
            y = y_raw

        # Align features with these rows using the cleaned X.
        X_subset = X_clean.loc[df_target.index].copy()

        # Encode categoricals for RF / LightGBM / XGBoost.
        X_enc = encode_categoricals_catboost(
            X=X_subset,
            y=y,
            cat_cols=cat_cols_global,
        )

        # Prepare CatBoost input (original features with same rows).
        X_cb = X_subset.copy()

        # Compute per-model importances for this target.
        df_imp = compute_model_importances_for_target(
            X_enc=X_enc,
            y=y,
            original_X_for_cb=X_cb,
            cat_cols_cb=cat_cols_global,
        )

        # Compute per-target mean_rank across models.
        rank_df = df_imp.rank(method="average", ascending=False)
        mean_rank = rank_df.mean(axis=1)
        df_imp_with_rank = df_imp.copy()
        df_imp_with_rank["mean_rank"] = mean_rank

        # Sort and save per-target CSV.
        df_imp_sorted = df_imp_with_rank.sort_values(
            "mean_rank",
            ascending=True,
        )
        out_path_target = OUTPUT_DIR / f"feature_importances_{y_col}.csv"
        df_imp_sorted.to_csv(out_path_target, index=True)
        logging.info("Saved per-target feature importances to %s", out_path_target)

        # Update global aggregation.
        global_rank_sum = global_rank_sum.add(
            mean_rank.reindex(feature_names).fillna(0.0),
            fill_value=0.0,
        )
        global_rank_count += 1

    if global_rank_count == 0:
        logging.warning(
            "No targets processed (all had fewer than %d labelled rows). "
            "Global ranking will not be created.",
            MIN_SAMPLES_PER_TARGET,
        )
        return

    # Compute global mean rank across all processed targets.
    global_mean_rank = global_rank_sum / float(global_rank_count)
    df_global = pd.DataFrame({"global_mean_rank": global_mean_rank})
    df_global_sorted = df_global.sort_values("global_mean_rank", ascending=True)

    out_path_global = OUTPUT_DIR / "feature_importances_global_all_targets.csv"
    df_global_sorted.to_csv(out_path_global, index=True)
    logging.info(
        "Saved global feature ranking across %d targets to %s",
        global_rank_count,
        out_path_global,
    )

    # Log top-K globally important features.
    top_features_global = df_global_sorted.head(TOP_K_FEATURES).index.tolist()
    logging.info("Global top %d features across all targets:", TOP_K_FEATURES)
    for feature in top_features_global:
        logging.info("  %s", feature)


if __name__ == "__main__":
    main()
