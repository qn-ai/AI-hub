#!/usr/bin/env python
"""Feature reduction using tree-based model importances.

This script:
- Loads a CSV dataset.
- Detects `id_`, `ft_`, and `y_` columns by prefix.
- Reduces features via:
    * high-missing-column removal,
    * zero-variance removal,
    * high-correlation removal.
- Encodes categoricals with CatBoostEncoder (no rare-category combining).
- Trains RandomForest, LightGBM, CatBoost, and XGBoost.
- Combines feature importances across models using rank aggregation.
- Saves combined importances to `feature_importances_<target>.csv`.
- Logs the top-k features and resulting reduced feature matrix shape.

Usage:
    Adjust the CONFIG section, then run:

        python feature_reduction.py
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier

# =====================================================================
# CONFIG
# =====================================================================

DATA_PATH = "input_data.csv"  # Path to input CSV file.
TARGET_COL = "y_target"  # One target column name, e.g. "y_abc".

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

MISSING_THRESH = 0.8  # Drop features with > 80% missing.
CORR_THRESH = 0.95  # Drop numeric features with |corr| > 0.95.
TOP_K_FEATURES = 100  # Number of top features to select.
RANDOM_STATE = 42

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
    """Perform basic feature reduction.

    Operations:
      * Drop columns with too many missing values.
      * Drop near-zero variance numeric features.
      * Drop highly correlated numeric features.

    Args:
        X: Feature matrix.
        missing_thresh: Threshold for maximum allowed missing fraction.
        corr_thresh: Threshold for absolute correlation to drop features.

    Returns:
        A reduced feature matrix after cleanup.
    """
    logging.info("Initial features: %d", X.shape[1])

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
    cat_cols = [col for col in X.columns if col not in num_cols]

    # 2) Drop near-zero variance numeric features.
    if num_cols:
        vt = VarianceThreshold(threshold=0.0)
        X_num = X[num_cols]
        vt.fit(X_num)
        keep_mask = vt.get_support()
        keep_num_cols = [
            col for col, keep in zip(num_cols, keep_mask, strict=True) if keep
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

    logging.info("Features after basic cleanup: %d", X.shape[1])
    # Recompute cat_cols only to avoid unused variable warnings.
    _ = [col for col in X.columns if col not in X.select_dtypes(include=[np.number])]
    return X


def encode_categoricals_catboost(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Encode categorical features using CatBoostEncoder.

    This does **not** combine rare categories. All category levels are
    preserved as-is and encoded into numeric features.

    Args:
        X: Feature matrix after basic cleanup.
        y: Target series aligned with ``X``.

    Returns:
        A tuple of:
        - X_enc: Encoded feature matrix (all numeric).
        - num_cols: Names of numeric columns in the original ``X``.
        - cat_cols: Names of categorical columns in the original ``X``.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]

    if cat_cols:
        logging.info(
            "Encoding %d categorical columns with CatBoostEncoder (no rare "
            "category combining)",
            len(cat_cols),
        )
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        X_enc = encoder.fit_transform(X, y)
        return X_enc, num_cols, cat_cols

    logging.info("No categorical columns detected; skipping encoding")
    return X.copy(), num_cols, cat_cols


def get_xgb_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Train XGBoost and compute feature importances (gain).

    Args:
        X: Encoded feature matrix (all numeric).
        y: Target series.

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


def compute_model_importances(
    X_enc: pd.DataFrame,
    y: pd.Series,
    original_X_for_cb: pd.DataFrame,
    cat_cols_cb: List[str],
) -> pd.DataFrame:
    """Compute feature importances from RF, LightGBM, CatBoost and XGBoost.

    Args:
        X_enc:
            Encoded feature matrix (numeric) for RandomForest, LightGBM,
            and XGBoost.
        y:
            Target series.
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

    # CatBoost (native categorical handling, unchanged categories).
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


def combine_importances(
    df_imp: pd.DataFrame,
    top_k: int = TOP_K_FEATURES,
) -> Tuple[pd.DataFrame, List[str]]:
    """Combine feature importances using rank aggregation.

    Ranks features separately for each model and then averages the ranks.

    Args:
        df_imp:
            DataFrame of feature importances with one column per model.
        top_k:
            Number of top features to return.

    Returns:
        A tuple of:
        - df_imp_sorted: DataFrame sorted by increasing mean_rank.
        - top_k_features: List of the top-k feature names.
    """
    rank_df = df_imp.rank(method="average", ascending=False)
    df_imp["mean_rank"] = rank_df.mean(axis=1)

    df_imp_sorted = df_imp.sort_values("mean_rank", ascending=True)
    top_features = df_imp_sorted.head(top_k).index.tolist()
    return df_imp_sorted, top_features


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    """Run the feature reduction pipeline."""
    logging.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    id_cols, ft_cols, y_cols = detect_columns(df)
    logging.info(
        "Detected %d id_, %d ft_, %d y_ columns",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if TARGET_COL not in df.columns:
        msg = f"TARGET_COL '{TARGET_COL}' not found in dataframe"
        raise ValueError(msg)

    # Keep rows where target is not missing.
    df_target = df[df[TARGET_COL].notna()].copy()
    logging.info("Rows with non-missing %s: %d", TARGET_COL, df_target.shape[0])

    X_raw = df_target[ft_cols].copy()
    y = df_target[TARGET_COL]

    # 1) Basic feature cleanup.
    X_clean = basic_feature_cleanup(X_raw)

    # 2) Encode categoricals for RF / LightGBM / XGBoost (no rare merging).
    X_enc, num_cols, cat_cols = encode_categoricals_catboost(X_clean, y)
    logging.info(
        "After encoding: %d numeric + %d encoded categorical columns",
        len(num_cols),
        len(cat_cols),
    )

    # 3) Prepare CatBoost input (original features after cleanup).
    X_cb = X_clean.copy()

    # 4) Compute importances from all four models.
    df_imp = compute_model_importances(
        X_enc=X_enc,
        y=y,
        original_X_for_cb=X_cb,
        cat_cols_cb=cat_cols,
    )

    # 5) Combine and select top-k features.
    df_imp_sorted, top_features = combine_importances(
        df_imp,
        top_k=TOP_K_FEATURES,
    )

    out_path = f"feature_importances_{TARGET_COL}.csv"
    df_imp_sorted.to_csv(out_path, index=True)
    logging.info("Saved combined feature importances to %s", out_path)

    logging.info("Top %d features:", TOP_K_FEATURES)
    for feature in top_features:
        logging.info("  %s", feature)

    # Reduced feature matrix for downstream modelling (if needed).
    X_reduced = X_enc[top_features].copy()
    logging.info("Reduced feature matrix shape: %s", X_reduced.shape)


if __name__ == "__main__":
    main()
