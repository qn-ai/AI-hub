#!/usr/bin/env python
"""
Stage-2: Per-Target Model Training with Dynamic CV Folds and
Saving One Trained Model Per Algorithm (RF, LGBM, XGB, HGB, CB).

This stage:

- Uses feature_importances_<y>.csv from Stage-1 to choose features per target.
- For each y_*:
    * Filters non-missing rows
    * Checks class distribution and chooses n_splits = min(MAX_N_SPLITS, min_class_count)
    * Skips targets with too few rows or degenerate classes
    * Builds two feature views:
        - Numeric (CatBoostEncoder) for RF / LGBM / XGB / HGB
        - Raw string categorical view for CatBoost
    * Cross-validates all 5 models and computes metrics:
        - F1, Precision, Recall, Accuracy, AUC
    * Selects the best model by F1
    * Refits ALL 5 models on the full target data
    * Saves:
        - trained_models/y_<target>_RF.joblib
        - trained_models/y_<target>_LGBM.joblib
        - trained_models/y_<target>_XGB.joblib
        - trained_models/y_<target>_HGB.joblib
        - trained_models/y_<target>_CB.joblib
        - trained_models/y_<target>_best.joblib (alias to best model)

Outputs:

- model_cv_results_parallel.csv
- model_cv_results_parallel.json
- skipped_targets_stage2.csv
- logs/y_<target>_stage2.log
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

# dynamic CV folds: n_splits_target = min(MAX_N_SPLITS, min_class_count)
MAX_N_SPLITS = 5

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

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage2_file", False)
        for h in logger.handlers
    ):
        fh = logging.FileHandler(LOG_DIR / f"{y_col}_stage2.log", mode="w", encoding="utf-8")
        fh._stage2_file = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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


def load_selected_features(y_col: str, logger: logging.Logger) -> List[str]:
    """Load selected features from Stage-1 importance file.

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
    """Encode target and determine n_splits for CV.

    - Convert to string
    - Compute class counts
    - If min class count < 2 → skip
    - n_splits_target = min(max_splits, min_class_count)
    - LabelEncoder for contiguous 0..K-1
    """
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    min_count = int(counts.min())
    n_classes = counts.shape[0]

    logger.info("Class distribution: %s", counts.to_dict())

    if min_count < 2:
        logger.warning(
            "Skipping target: min class count = %d < 2 (n_classes=%d)",
            min_count,
            n_classes,
        )
        return None, None, None

    n_splits_target = int(min(max_splits, min_count))
    if n_splits_target < 2:
        logger.warning(
            "Skipping target: computed n_splits=%d < 2", n_splits_target
        )
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
    """Prepare numeric view using CatBoostEncoder (unsupervised)."""
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        dummy = np.zeros(len(X_num))
        X_num = enc.fit_transform(X_num, dummy)
    elif not USE_CATBOOST_ENCODER and cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    return X_num


def build_models(is_binary: bool) -> Dict[str, object]:
    """Return model prototypes."""
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    is_binary: bool,
) -> Dict[str, float]:
    """Compute metrics for one fold."""
    avg = "binary" if is_binary else "macro"

    out: Dict[str, float] = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["f1"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    out["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)

    if y_proba is None:
        out["auc"] = np.nan
        return out

    try:
        if is_binary:
            if y_proba.ndim == 1:
                out["auc"] = roc_auc_score(y_true, y_proba)
            else:
                out["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except Exception:
        out["auc"] = np.nan

    return out


def cross_validate_target(
    y_col: str,
    X: pd.DataFrame,
    y_enc: pd.Series,
    n_splits_target: int,
    logger: logging.Logger,
) -> Tuple[Optional[str], Dict[str, Dict[str, float]], Dict[str, object]]:
    """
    Cross-validate all models for a target, return:

    - best_model_name (by F1),
    - metrics per model,
    - dict of trained models (refit on full data for each algorithm).
    """
    is_binary = (y_enc.nunique() == 2)
    logger.info("Target %s is %s", y_col, "binary" if is_binary else "multiclass")

    cv = StratifiedKFold(
        n_splits=n_splits_target,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Feature views
    X_num = prepare_numeric_view_with_encoder(X)
    X_cb, cat_cols = prepare_catboost_view(X)
    cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

    models_proto = build_models(is_binary)
    all_results: Dict[str, Dict[str, float]] = {}
    best_model_name: Optional[str] = None
    best_f1 = -np.inf

    # ------ CV for each model ------
    for name, model_p in models_proto.items():
        logger.info("CV for model: %s", name)
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_num, y_enc), start=1):
            logger.info("  Fold %d/%d", fold_idx, n_splits_target)
            if name == "CB":
                X_train = X_cb.iloc[train_idx]
                X_val = X_cb.iloc[val_idx]
            else:
                X_train = X_num.iloc[train_idx]
                X_val = X_num.iloc[val_idx]

            y_train = y_enc.iloc[train_idx]
            y_val = y_enc.iloc[val_idx]

            model = model_p.__class__(**model_p.get_params())
            if name == "CB":
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_indices if cat_indices else None,
                )
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            y_proba: Optional[np.ndarray] = None
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
            fold_metrics.append(m)

        # aggregate
        agg = {
            k: float(np.nanmean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0].keys()
        }
        agg["n_splits"] = float(n_splits_target)
        all_results[name] = agg
        logger.info("  CV metrics for %s: %s", name, agg)

        if agg["f1"] > best_f1:
            best_f1 = agg["f1"]
            best_model_name = name

    if best_model_name is None:
        logger.warning("No valid model for %s", y_col)
        return None, all_results, {}

    logger.info("Best model for %s is %s (F1=%.4f)", y_col, best_model_name, best_f1)

    # ------ Refit ALL models on full data ------
    trained_models: Dict[str, object] = {}
    models_proto_full = build_models(is_binary)

    for name, model_p in models_proto_full.items():
        logger.info("Refitting full model for %s", name)
        model = model_p.__class__(**model_p.get_params())
        if name == "CB":
            model.fit(
                X_cb,
                y_enc,
                cat_features=cat_indices if cat_indices else None,
            )
        else:
            X_num_full = prepare_numeric_view_with_encoder(X)
            model.fit(X_num_full, y_enc)
        trained_models[name] = model

    return best_model_name, all_results, trained_models


# ---------------------------------------------------------------------
# PER-TARGET PIPELINE
# ---------------------------------------------------------------------

def process_target(y_col: str, df: pd.DataFrame) -> Dict:
    logger = get_target_logger(y_col)
    logger.info("=== Stage-2 training for %s ===", y_col)

    df_target = df[df[y_col].notna()].copy()
    n_rows = df_target.shape[0]

    if n_rows < 2:
        logger.warning("Skipping %s: <2 labelled rows.", y_col)
        return {"target": y_col, "skipped": True, "reason": "too_few_rows"}

    selected_features = load_selected_features(y_col, logger)
    if not selected_features:
        logger.warning("Skipping %s: no selected features.", y_col)
        return {"target": y_col, "skipped": True, "reason": "no_selected_features"}

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        logger.warning(
            "Skipping %s: missing features in data: %s", y_col, missing
        )
        return {"target": y_col, "skipped": True, "reason": "missing_features"}

    X = df_target[selected_features].copy()
    y_raw = df_target[y_col]

    y_enc, le, n_splits_target = prepare_target_and_cv(
        y_raw=y_raw,
        max_splits=MAX_N_SPLITS,
        logger=logger,
    )
    if y_enc is None:
        return {"target": y_col, "skipped": True, "reason": "too_few_classes"}

    best_model_name, all_results, trained_models = cross_validate_target(
        y_col=y_col,
        X=X,
        y_enc=y_enc,
        n_splits_target=n_splits_target,
        logger=logger,
    )

    if not trained_models or best_model_name is None:
        logger.warning("No trained models for %s", y_col)
        return {"target": y_col, "skipped": True, "reason": "no_valid_model"}

    # ------ Save all models ------
    for model_name, est in trained_models.items():
        model_path = TRAINED_MODELS_DIR / f"{y_col}_{model_name}.joblib"
        joblib.dump(est, model_path)
        logger.info("Saved %s model for %s to %s", model_name, y_col, model_path)

    # alias best model
    best_model = trained_models[best_model_name]
    best_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    joblib.dump(best_model, best_path)
    logger.info("Saved BEST model for %s (%s) to %s", y_col, best_model_name, best_path)

    # Flatten metrics for global table
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
        log.error("No y_ columns found.")
        return

    log.info(
        "Starting Stage-2 training for %d targets with N_JOBS_TARGETS=%d",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(process_target)(y_col, df) for y_col in y_cols
    )

    processed = [r for r in results if not r.get("skipped", False)]
    skipped = [r for r in results if r.get("skipped", False)]

    if processed:
        df_res = pd.DataFrame(processed)
        df_res.to_csv(RESULTS_CSV, index=False)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        log.info("Saved Stage-2 metrics to %s and %s", RESULTS_CSV, RESULTS_JSON)
    else:
        log.warning("No targets successfully trained in Stage-2.")

    if skipped:
        df_skip = pd.DataFrame(skipped)
        df_skip.to_csv(SKIPPED_CSV, index=False)
        log.info("Saved skipped targets list to %s", SKIPPED_CSV)


if __name__ == "__main__":
    main()
