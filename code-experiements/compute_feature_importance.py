#!/usr/bin/env python
"""Multi-target feature importance using tree-based models (RF, LGBM, XGB, CB, HGB).

Assumes a wide table with:
- identifier columns starting with ``id_``,
- feature columns starting with ``ft_``,
- target columns starting with ``y_``.

Pipeline:
- Load the CSV dataset.
- Detect id_, ft_, y_ columns by prefix.
- GLOBAL (once, all rows, ft_ only) [optional]:
    * drop zero-variance numeric columns,
    * drop highly-correlated numeric columns (|corr| > CORR_THRESH).
- For each target column (y_*) with enough non-missing rows:
    * subset rows where this y_ is not missing,
    * per-target [optional]: drop ft_ columns with > MISSING_THRESH missing
      (based only on rows available for that target),
    * label-encode the target if it is non-numeric,
    * detect binary vs multiclass (n_unique(y)),
    * treat ALL object columns as categorical,
    * encode categoricals with CatBoostEncoder (no rare-category combining)
      for RF / LGBM / XGB / HGB (optional toggle),
    * coerce all encoded features to numeric; NaNs are allowed
      (RF can optionally see a median-imputed copy),
    * for CatBoost:
        - use original features,
        - convert categorical columns to string and fill NaNs with "NA_CAT",
        - choose loss_function="Logloss" for binary, "MultiClass" for multiclass,
    * train:
        - RandomForest (on encoded numeric features; optional median impute),
        - LightGBM (NaN-aware),
        - XGBoost (NaN-aware, objective chosen by binary/multiclass),
        - CatBoost (NaN-aware, string categoricals),
        - HistGradientBoostingClassifier (NaN-aware, encoded features),
    * compute feature importances for that target,
    * aggregate model importances into a per-target mean_rank,
    * save combined file:
        ``feature_importances_<y_col>.csv``,
    * optionally save per-model ranking files:
        ``feature_importances_<y_col>_RF.csv``,
        ``..._LGBM.csv``, ``..._CB.csv``, ``..._XGB.csv``, ``..._HGB.csv``.
- Optionally aggregate per-target mean_rank across all processed y_ columns
  into a global ranking and save as
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =====================================================================
# CONFIG + TOGGLES
# =====================================================================

DATA_PATH = "input_data.csv"  # Path to input CSV file.

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

MISSING_THRESH = 0.8  # Drop features with > 80% missing (per target).
CORR_THRESH = 0.95  # Drop numeric features with |corr| > 0.95 (global).
MIN_SAMPLES_PER_TARGET = 200  # Skip targets with fewer labelled rows.

RANDOM_STATE = 42

OUTPUT_DIR = Path("feature_importances")  # Folder for CSV outputs.

# ---- Optional toggles ----
USE_GLOBAL_VAR_CORR_CLEANUP = True          # global variance + correlation cleanup
USE_PER_TARGET_MISSING_CLEANUP = True       # drop high-missing ft_ per target
USE_CATBOOST_ENCODER = True                 # encode object cols for RF/LGBM/XGB/HGB
RF_MEDIAN_IMPUTE_NUMERIC = True             # median-impute numeric NaNs for RF only
SAVE_PER_MODEL_FILES = False                # write _RF/_LGBM/_CB/_XGB/_HGB CSVs
SAVE_GLOBAL_RANKING = True                  # write global ranking CSV

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
    """Detect identifier, feature, and target columns by prefix."""
    id_cols = [col for col in df.columns if col.startswith(ID_PREFIX)]
    ft_cols = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    y_cols = [col for col in df.columns if col.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def global_feature_cleanup(
    X: pd.DataFrame,
    corr_thresh: float = CORR_THRESH,
) -> pd.DataFrame:
    """Perform global unsupervised cleanup on ft_ columns (optional).

    - Drop zero-variance numeric features.
    - Drop highly correlated numeric features (|corr| > corr_thresh).
    """
    if not USE_GLOBAL_VAR_CORR_CLEANUP:
        logging.info(
            "Global cleanup: disabled by USE_GLOBAL_VAR_CORR_CLEANUP=False; "
            "keeping all ft_ columns as-is."
        )
        return X

    logging.info("Global cleanup: initial feature count: %d", X.shape[1])

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
    """Drop high-missing columns for a specific target's sample subset (optional)."""
    if not USE_PER_TARGET_MISSING_CLEANUP:
        logging.info(
            "Per-target missing cleanup disabled; keeping all features for this target."
        )
        return X

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
    """Encode categorical features using CatBoostEncoder (optional).

    Rules:
    - Any column with dtype "object" is treated as categorical.
    - If USE_CATBOOST_ENCODER=True, encode these categoricals.
    - If USE_CATBOOST_ENCODER=False, drop object columns (log a warning).
    - All columns are coerced to numeric; NaNs are allowed.
    """
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        logging.info(
            "Encoding %d categorical columns with CatBoostEncoder for this "
            "target: %s",
            len(cat_cols),
            cat_cols,
        )
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        X_enc = encoder.fit_transform(X, y)
    elif USE_CATBOOST_ENCODER and not cat_cols:
        logging.info("No categorical columns detected for this target")
        X_enc = X.copy()
    else:
        # Not using encoder: drop object columns.
        if cat_cols:
            logging.warning(
                "USE_CATBOOST_ENCODER=False -> dropping %d object columns: %s",
                len(cat_cols),
                cat_cols,
            )
        X_enc = X.drop(columns=cat_cols)

    # Ensure everything is numeric for RF / LGBM / XGB / HGB.
    X_enc = X_enc.apply(pd.to_numeric, errors="coerce")

    obj_after = X_enc.select_dtypes(include=["object"]).columns.tolist()
    if obj_after:
        logging.warning(
            "Dropping %d columns that remain non-numeric after encoding: %s",
            len(obj_after),
            obj_after,
        )
        X_enc = X_enc.drop(columns=obj_after)

    return X_enc, cat_cols


def get_xgb_importance(
    X: pd.DataFrame,
    y: pd.Series,
    is_binary: bool,
    n_classes: int,
) -> pd.Series:
    """Train XGBoost and compute feature importances (gain)."""
    logging.info("Training XGBoost for feature importance")

    if is_binary:
        objective = "binary:logistic"
        num_class = None
    else:
        objective = "multi:softprob"
        num_class = n_classes

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective=objective,
        eval_metric="logloss",
        num_class=num_class,
    )
    model.fit(X, y)
    booster = model.get_booster()
    raw_score = booster.get_score(importance_type="gain")

    importance = pd.Series(0.0, index=X.columns)

    for fname, score in raw_score.items():
        # Case 1: XGBoost returns actual column names.
        if fname in importance.index:
            importance[fname] = score
            continue

        # Case 2: legacy behavior "f0", "f1", ...
        if fname.startswith("f"):
            rest = fname[1:]
            if rest.isdigit():
                idx = int(rest)
                if idx < len(X.columns):
                    col_name = X.columns[idx]
                    importance[col_name] = score

    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance


def get_hgb_importance(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """Train HistGradientBoostingClassifier and compute permutation importance.

    HGB is native-NaN aware and works well on encoded numeric features.
    We use permutation_importance to get model-agnostic importances.
    """
    logging.info("Training HistGradientBoostingClassifier for feature importance")

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",  # works for binary and multiclass
        max_depth=None,
        learning_rate=0.05,
        max_bins=255,
        l2_regularization=0.0,
        random_state=RANDOM_STATE,
    )
    hgb.fit(X, y)

    result = permutation_importance(
        hgb,
        X,
        y,
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.Series(result.importances_mean, index=X.columns)
    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance


def compute_model_importances_for_target(
    X_enc: pd.DataFrame,
    y: pd.Series,
    X_cb: pd.DataFrame,
    cat_cols_cb: List[str],
    is_binary: bool,
    n_classes: int,
) -> pd.DataFrame:
    """Compute feature importances from RF, LGBM, XGB, CatBoost, and HGB."""
    feature_names = X_enc.columns

    # ------------------------------------------------------------------
    # Random Forest (on encoded features; optional median impute for NaNs).
    # ------------------------------------------------------------------
    logging.info("Training RandomForest for feature importance")
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if RF_MEDIAN_IMPUTE_NUMERIC:
        X_rf = X_enc.copy()
        num_cols = X_rf.columns
        medians = X_rf[num_cols].median()
        X_rf[num_cols] = X_rf[num_cols].fillna(medians)
        rf.fit(X_rf, y)
    else:
        rf.fit(X_enc, y)

    imp_rf = pd.Series(rf.feature_importances_, index=feature_names)

    # ------------------------------------------------------------------
    # LightGBM (NaN-aware, objective based on binary/multiclass).
    # ------------------------------------------------------------------
    logging.info("Training LightGBM for feature importance")
    if is_binary:
        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="binary",
        )
    else:
        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="multiclass",
            num_class=n_classes,
        )
    lgbm.fit(X_enc, y)
    imp_lgb = pd.Series(lgbm.feature_importances_, index=feature_names)

    # ------------------------------------------------------------------
    # XGBoost (NaN-aware, objective based on binary/multiclass).
    # ------------------------------------------------------------------
    imp_xgb = get_xgb_importance(
        X_enc,
        y,
        is_binary=is_binary,
        n_classes=n_classes,
    )

    # ------------------------------------------------------------------
    # CatBoost (native categorical handling on X_cb).
    # ------------------------------------------------------------------
    logging.info("Training CatBoost for feature importance")
    loss_fn = "Logloss" if is_binary else "MultiClass"

    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function=loss_fn,
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

    # ------------------------------------------------------------------
    # HistGradientBoostingClassifier (NaN-aware) + permutation importance.
    # ------------------------------------------------------------------
    imp_hgb = get_hgb_importance(X_enc, y)

    df_imp = pd.DataFrame(
        {
            "RF": imp_rf,
            "LGBM": imp_lgb,
            "CB": imp_cb_aligned,
            "XGB": imp_xgb,
            "HGB": imp_hgb.reindex(feature_names).fillna(0.0),
        }
    ).fillna(0.0)

    return df_imp


def save_per_model_rankings(
    y_col: str,
    df_imp: pd.DataFrame,
) -> None:
    """Save per-model feature rankings to separate CSV files (optional)."""
    if not SAVE_PER_MODEL_FILES:
        logging.info(
            "SAVE_PER_MODEL_FILES=False -> skipping per-model CSV exports "
            "for target %s",
            y_col,
        )
        return

    for model_name in ["RF", "LGBM", "CB", "XGB", "HGB"]:
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
    """Run the multi-target feature importance pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    id_cols, ft_cols, y_cols = detect_columns(df)
    logging.info(
        "Detected %d id_, %d ft_, %d y_ columns",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not ft_cols:
        raise ValueError("No feature columns (ft_*) detected.")
    if not y_cols:
        raise ValueError("No target columns (y_*) detected.")

    # GLOBAL: unsupervised cleanup on all ft_ columns (zero-variance + corr).
    X_all = df[ft_cols].copy()
    X_base = global_feature_cleanup(X_all)
    base_feature_names = X_base.columns

    # For global aggregation across targets (over mean_rank).
    global_rank_sum = pd.Series(0.0, index=base_feature_names)
    global_rank_count = pd.Series(0, index=base_feature_names, dtype="int64")

    # Loop over all y_ targets.
    for y_col in y_cols:
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

        if not np.issubdtype(y_raw.dtype, np.number):
            logging.info("Label-encoding non-numeric target for %s", y_col)
            le = LabelEncoder()
            y = pd.Series(
                le.fit_transform(y_raw.astype(str)),
                index=y_raw.index,
            )
        else:
            y = y_raw.copy()

        y = y.astype("int64")

        # Binary vs multiclass check.
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            logging.info(
                "Skipping %s: target has only one class after encoding.",
                y_col,
            )
            continue

        is_binary = n_classes == 2
        logging.info(
            "Target %s: %d classes detected (%s)",
            y_col,
            n_classes,
            "binary" if is_binary else "multiclass",
        )

        # Align global-cleaned features with these rows.
        X_subset = X_base.loc[df_target.index].copy()

        # PER-TARGET: drop features with >80% missing (only on these rows).
        X_target = per_target_missing_cleanup(X_subset)

        if X_target.shape[1] == 0:
            logging.info(
                "Skipping %s: all features dropped by per-target missing filter",
                y_col,
            )
            continue

        # Identify categorical columns for CatBoost on ORIGINAL features.
        cat_cols_t = X_target.select_dtypes(include=["object"]).columns.tolist()

        # Encode categoricals for RF / LGBM / XGB / HGB (keep numeric NaNs).
        X_enc, _ = encode_categoricals_catboost(X=X_target, y=y)

        # Prepare CatBoost input: categorical as strings, NaNs -> "NA_CAT".
        X_cb = X_target.copy()
        for c in cat_cols_t:
            X_cb[c] = X_cb[c].astype("string").fillna("NA_CAT")

        # Compute per-model importances for this target.
        df_imp = compute_model_importances_for_target(
            X_enc=X_enc,
            y=y,
            X_cb=X_cb,
            cat_cols_cb=cat_cols_t,
            is_binary=is_binary,
            n_classes=n_classes,
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

        # Optional per-model CSVs.
        save_per_model_rankings(y_col, df_imp)

        # Update global aggregation only for features that exist for this target.
        aligned_rank = mean_rank.reindex(base_feature_names)
        valid_mask = aligned_rank.notna()
        global_rank_sum.loc[valid_mask] += aligned_rank.loc[valid_mask]
        global_rank_count.loc[valid_mask] += 1

    # After all targets processed, compute global ranking (optional).
    if not SAVE_GLOBAL_RANKING:
        logging.info("SAVE_GLOBAL_RANKING=False -> skipping global ranking CSV.")
        return

    if not (global_rank_count > 0).any():
        logging.warning(
            "No targets processed (all had fewer than %d labelled rows "
            "or all features dropped). Global ranking will not be created.",
            MIN_SAMPLES_PER_TARGET,
        )
        return

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


if __name__ == "__main__":
    main()
