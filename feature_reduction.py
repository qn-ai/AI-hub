#!/usr/bin/env python
"""Multi-target feature reduction using tree-based model importances.

Assumes a wide table with:
- identifier columns starting with ``id_``,
- feature columns starting with ``ft_``,
- target columns starting with ``y_``.

Pipeline:
- Load the CSV dataset.
- Detect id_, ft_, y_ columns by prefix.
- GLOBAL (once, all rows, ft_ only):
    * drop zero-variance numeric columns,
    * drop highly-correlated numeric columns (|corr| > CORR_THRESH).
    (Missing-value filtering is NOT done globally.)
- For each target column (y_*) with enough non-missing rows:
    * subset rows where this y_ is not missing,
    * per-target: drop ft_ columns with > MISSING_THRESH missing
      (based only on rows available for that target),
    * label-encode the target if it is non-numeric,
    * encode categoricals with CatBoostEncoder (no rare-category combining),
    * coerce all encoded features to numeric (but DO NOT fill NaNs),
    * train:
        - RandomForest (sklearn>=1.4, NaN-aware),
        - LightGBM (NaN-aware),
        - XGBoost (NaN-aware),
        - CatBoost (NaN-aware, native categorical handling),
    * compute feature importances for that target,
    * aggregate model importances into a per-target mean_rank,
    * save:
        - combined file: ``feature_importances_<y_col>.csv``,
        - per-model ranking files:
          ``feature_importances_<y_col>_RF.csv``,
          ``..._LGBM.csv``, ``..._CB.csv``, ``..._XGB.csv``.
- Aggregate per-target mean_rank across all processed y_ columns into a
  global ranking and save as
  ``feature_importances_global_all_targets.csv``.
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

MISSING_THRESH = 0.8  # Drop features with > 80% missing (per target).
CORR_THRESH = 0.95  # Drop numeric features with |corr| > 0.95 (global).
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


def global_feature_cleanup(
    X: pd.DataFrame,
    corr_thresh: float = CORR_THRESH,
) -> pd.DataFrame:
    """Perform global unsupervised cleanup on ft_ columns.

    Operations (same for all targets, using all rows):
      * Drop zero-variance numeric features.
      * Drop highly correlated numeric features (|corr| > corr_thresh).

    Missing-value filtering is NOT done here; it is done per-target.

    Args:
        X: Feature matrix with ft_ columns only (all rows).
        corr_thresh: Threshold for absolute correlation to drop features.

    Returns:
        A reduced feature matrix after global cleanup (same rows, fewer cols).
    """
    logging.info("Global cleanup: initial feature count: %d", X.shape[1])

    # Identify numeric vs categorical.
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 1) Drop zero-variance numeric features.
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
                "Global cleanup: dropping %d numeric columns with zero variance",
                len(drop_var),
            )
            X = X.drop(columns=drop_var)
            num_cols = keep_num_cols

    # 2) Drop highly correlated numeric features.
    if len(num_cols) > 1:
        corr = X[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        if to_drop:
            logging.info(
                "Global cleanup: dropping %d numeric columns with corr > %.2f",
                len(to_drop),
                corr_thresh,
            )
            X = X.drop(columns=to_drop)

    logging.info("Global cleanup: feature count after cleanup: %d", X.shape[1])
    return X


def per_target_missing_cleanup(
    X: pd.DataFrame,
    missing_thresh: float = MISSING_THRESH,
) -> pd.DataFrame:
    """Drop high-missing columns for a specific target's sample subset.

    The missing ratio is computed using ONLY the rows available for that
    target. Different targets can drop different feature columns.

    Args:
        X: Feature matrix subset for a single target (rows where y is not NaN).
        missing_thresh: Threshold for maximum allowed missing fraction.

    Returns:
        A reduced feature matrix for this target.
    """
    missing_ratio = X.isna().mean()
    drop_missing = missing_ratio[missing_ratio > missing_thresh].index.tolist()
    if drop_missing:
        logging.info(
            "Per-target cleanup: dropping %d columns with missing_ratio > %.2f",
            len(drop_missing),
            missing_thresh,
        )
        X = X.drop(columns=drop_missing)
    return X


def encode_categoricals_catboost(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    """Encode categorical features using CatBoostEncoder.

    This does **not** combine rare categories. All category levels are
    preserved as-is and encoded into numeric features.

    Also:
    - coerces all columns to numeric,
    - leaves NaNs as-is (no imputation).

    Args:
        X: Feature matrix after per-target cleanup (subset of rows & cols).
        y: Target series aligned with ``X`` (numeric / label-encoded).

    Returns:
        A tuple of:
        - X_enc: Encoded, numeric feature matrix (may contain NaNs).
        - cat_cols: Names of categorical columns that were encoded.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]

    if cat_cols:
        logging.info(
            "Encoding %d categorical columns with CatBoostEncoder for this "
            "target",
            len(cat_cols),
        )
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        X_enc = encoder.fit_transform(X, y)
    else:
        logging.info("No categorical columns detected for this target")
        X_enc = X.copy()

    # Ensure everything is numeric for RF / LGBM / XGB.
    X_enc = X_enc.apply(pd.to_numeric, errors="coerce")

    return X_enc, cat_cols


def get_xgb_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Train XGBoost and compute feature importances (gain).

    Args:
        X: Encoded feature matrix (all numeric; NaNs allowed).
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
    X_cb: pd.DataFrame,
    cat_cols_cb: List[str],
) -> pd.DataFrame:
    """Compute feature importances from RF, LightGBM, CatBoost, and XGBoost.

    All models here support NaN values:
      - RandomForestClassifier (sklearn>=1.4)
      - LightGBM
      - XGBoost
      - CatBoost

    Args:
        X_enc:
            Encoded feature matrix (numeric; may contain NaNs) for
            RandomForest, LightGBM, and XGBoost.
        y:
            Target series for the current y_ column (numeric / label-encoded).
        X_cb:
            Original (non-encoded) feature matrix for this target
            (same rows and columns as X_enc), for CatBoost.
        cat_cols_cb:
            Names of columns in X_cb treated as categorical by CatBoost.

    Returns:
        DataFrame with columns ``['RF', 'LGBM', 'CB', 'XGB']`` and features
        as the index, containing per-model importances.
    """
    feature_names = X_enc.columns

    # Random Forest
    logging.info("Training RandomForest (sklearn>=1.4, NaN-aware)")
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

    # CatBoost (native categorical handling on X_cb).
    logging.info("Training CatBoost for feature importance")
    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",  # For multiclass. Use 'Logloss' for binary.
        random_state=RANDOM_STATE,
        verbose=False,
    )
    if cat_cols_cb:
        cat_indices = [X_cb.columns.get_loc(col) for col in cat_cols_cb]
    else:
        cat_indices = None

    cb.fit(X_cb, y, cat_features=cat_indices)

    imp_cb_raw = cb.get_feature_importance()
    imp_cb = pd.Series(imp_cb_raw, index=X_cb.columns)
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


def save_per_model_rankings(
    y_col: str,
    df_imp: pd.DataFrame,
) -> None:
    """Save per-model feature rankings to separate CSV files.

    For each model column in df_imp (RF, LGBM, CB, XGB), create a CSV with
    columns:
        - feature
        - importance
        - rank
    sorted by ascending rank (most important first).

    Files are named:
        feature_importances_<y_col>_<model>.csv

    Args:
        y_col: Name of the current target column.
        df_imp: DataFrame of importances with columns RF, LGBM, CB, XGB.
    """
    for model_name in ["RF", "LGBM", "CB", "XGB"]:
        if model_name not in df_imp.columns:
            continue
        s = df_imp[model_name]
        ranks = s.rank(ascending=False, method="average")
        out_df = pd.DataFrame(
            {
                "feature": s.index,
                "importance": s.values,
                "rank": ranks.values,
            }
        ).sort_values("rank", ascending=True)
        out_path = OUTPUT_DIR / f"feature_importances_{y_col}_{model_name}.csv"
        out_df.to_csv(out_path, index=False)
        logging.info("Saved %s ranking for %s to %s", model_name, y_col, out_path)


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    """Run the multi-target feature reduction pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data from %s", DATA_PATH)
    # low_memory=False for more stable dtype inference on wide tables.
    df = pd.read_csv(DATA_PATH, low_memory=False)

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

    # GLOBAL: unsupervised cleanup on all ft_ columns (zero-variance + corr).
    X_all = df[ft_cols].copy()
    X_base = global_feature_cleanup(X_all)

    # Global feature index, used for global aggregation.
    base_feature_names = X_base.columns

    # For global aggregation across targets (over mean_rank).
    global_rank_sum = pd.Series(0.0, index=base_feature_names)
    global_rank_count = pd.Series(0, index=base_feature_names, dtype="int64")

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
            y = y_raw.copy()

        # Ensure y is integer type for all models.
        y = y.astype("int64")

        # Align global-cleaned features with these rows.
        X_subset = X_base.loc[df_target.index].copy()

        # PER-TARGET: drop features with >80% missing (only on these rows).
        X_target = per_target_missing_cleanup(X_subset)

        # If we dropped everything, skip.
        if X_target.shape[1] == 0:
            logging.info(
                "Skipping %s: all features dropped by per-target missing filter",
                y_col,
            )
            continue

        # Identify categorical columns for this target (for CatBoost).
        num_cols_t = X_target.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_t = [col for col in X_target.columns if col not in num_cols_t]

        # Encode categoricals for RF / LGBM / XGB (keep NaNs).
        X_enc, _ = encode_categoricals_catboost(
            X=X_target,
            y=y,
        )

        # For CatBoost, use original features (with same rows/cols as X_target).
        X_cb = X_target.copy()

        # Compute per-model importances for this target.
        df_imp = compute_model_importances_for_target(
            X_enc=X_enc,
            y=y,
            X_cb=X_cb,
            cat_cols_cb=cat_cols_t,
        )

        # Compute per-target mean_rank across models.
        rank_df = df_imp.rank(method="average", ascending=False)
        mean_rank = rank_df.mean(axis=1)
        df_imp_with_rank = df_imp.copy()
        df_imp_with_rank["mean_rank"] = mean_rank

        # Sort and save per-target combined CSV.
        df_imp_sorted = df_imp_with_rank.sort_values(
            "mean_rank",
            ascending=True,
        )
        out_path_target = OUTPUT_DIR / f"feature_importances_{y_col}.csv"
        df_imp_sorted.to_csv(out_path_target, index=True)
        logging.info("Saved combined feature importances to %s", out_path_target)

        # Save per-model rankings as separate CSV files.
        save_per_model_rankings(y_col=y_col, df_imp=df_imp)

        # Update global aggregation only for features that exist for this target.
        aligned_rank = mean_rank.reindex(base_feature_names)

        valid_mask = aligned_rank.notna()
        global_rank_sum.loc[valid_mask] += aligned_rank.loc[valid_mask]
        global_rank_count.loc[valid_mask] += 1

    # After all targets processed, compute global ranking.
    processed_any = (global_rank_count > 0).any()
    if not processed_any:
        logging.warning(
            "No targets processed (all had fewer than %d labelled rows "
            "or all features dropped). Global ranking will not be created.",
            MIN_SAMPLES_PER_TARGET,
        )
        return

    # Avoid division by zero by masking where count == 0.
    valid_global = global_rank_count > 0
    global_mean_rank = pd.Series(np.inf, index=base_feature_names)
    global_mean_rank.loc[valid_global] = (
        global_rank_sum.loc[valid_global] / global_rank_count.loc[valid_global]
    )

    df_global = pd.DataFrame({"global_mean_rank": global_mean_rank})
    df_global_sorted = df_global.sort_values("global_mean_rank", ascending=True)

    out_path_global = OUTPUT_DIR / "feature_importances_global_all_targets.csv"
    df_global_sorted.to_csv(out_path_global, index=True)
    logging.info(
        "Saved global feature ranking across targets to %s",
        out_path_global,
    )

    # Log top-K globally important features.
    top_features_global = df_global_sorted.head(TOP_K_FEATURES).index.tolist()
    logging.info("Global top %d features across all targets:", TOP_K_FEATURES)
    for feature in top_features_global:
        logging.info("  %s", feature)


if __name__ == "__main__":
    main()
