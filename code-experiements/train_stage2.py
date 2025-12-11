#!/usr/bin/env python3
"""
Stage-2: Per-target model training with dynamic CV folds and
consistent classification/regression logic mirroring Stage-1.

This script:

- Uses feature_importances_<y>.csv from Stage-1 to choose features per target.
- Supports two global modes (matching Stage-1):

    TASK_MODE = "classification"
        * All y_* treated as classification targets.
        * For each y_*:
            - Filters non-missing rows.
            - Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
            - Skips targets with only one class or smallest class too small.
            - Loads feature_importances_<y>.csv and selects features where:
                RF > 0 & LGBM > 0 & CB > 0 & XGB > 0 & HGB > 0.
            - Builds two feature views:
                - Numeric (CatBoostEncoder) for RF / LGBM / XGB / HGB.
                - Raw string categorical view for CatBoost.
            - Chooses dynamic n_splits as:
                n_splits = min(MAX_N_SPLITS_CLASSIFICATION, min_class_count)
                (must be >= 2).
            - Cross-validates all 5 models and computes metrics:
                F1 (macro), Precision (macro), Recall (macro),
                Accuracy, AUC (binary or multiclass).
            - Selects best model by F1.
            - Refits ALL 5 models on full data.
            - Saves:
                trained_models/y_<target>_RF.joblib
                trained_models/y_<target>_LGBM.joblib
                trained_models/y_<target>_XGB.joblib
                trained_models/y_<target>_HGB.joblib
                trained_models/y_<target>_CB.joblib
                trained_models/y_<target>_best.joblib

    TASK_MODE = "regression"
        * All y_* treated as regression candidates.
        * For each y_*:
            - Filters non-missing rows.
            - Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
            - Skips non-numeric targets.
            - Skips numeric targets with too few unique values
              (<= REGRESSION_MIN_UNIQUE).
            - Loads feature_importances_<y>.csv and selects features where:
                RF_REG > 0.
            - Builds numeric CatBoostEncoder view.
            - Uses KFold with dynamic n_splits:
                n_splits = min(MAX_N_SPLITS_REGRESSION, n_samples)
                (must be >= 2).
            - Cross-validates one or more regression models (configurable):
                * RF_REG   → RandomForestRegressor
                * LGBM_REG → LGBMRegressor
                * XGB_REG  → XGBRegressor
                * HGB_REG  → HistGradientBoostingRegressor
                * CB_REG   → CatBoostRegressor
            - Computes metrics:
                RMSE, MAE, R2.
            - Selects best model by lowest RMSE.
            - Refits all selected regression models on full data.
            - Saves:
                trained_models/y_<target>_<MODEL>.joblib for each enabled model
                trained_models/y_<target>_best.joblib (best model alias)

Outputs:

- model_cv_results_parallel.csv
- model_cv_results_parallel.json
- skipped_targets_stage2.csv
- logs/y_<target>_stage2.log (per-target logging)

Optional:

- MLflow tracking per target (off by default).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier, CatBoostRegressor
from joblib import Parallel, delayed, dump, load
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

# Optional MLflow
try:
    import mlflow  # type: ignore[import]

    mlflow_available = True
except Exception:  # pragma: no cover - optional
    mlflow = None  # type: ignore[assignment]
    mlflow_available = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Global task mode (mirror Stage-1):
#   "classification" → all targets treated as classification
#   "regression"     → all targets treated as regression (numeric only)
TASK_MODE = "classification"  # or "regression"

data_path = "input_data.csv"
feature_importance_dir = Path("feature_importances")
models_dir = Path("trained_models")
log_dir = Path("logs")

id_prefix = "id_"
feature_prefix = "ft_"
target_prefix = "y_"

random_state = 42

min_samples_per_target = 200
min_class_count_for_training = 2  # classification

# Regression unique-values threshold (same as Stage-1)
REGRESSION_MIN_UNIQUE = 10

max_n_splits_classification = 5
max_n_splits_regression = 5

cpu_count = os.cpu_count() or 4
n_jobs_targets = max(min(cpu_count - 1, 16), 2)

use_catboost_encoder = True
cat_fill_value = "NA_CAT"

# Regression model choices (you can change this list)
# Allowed keys: "RF_REG", "LGBM_REG", "XGB_REG", "HGB_REG", "CB_REG"
REGRESSION_MODELS: List[str] = ["RF_REG"]

# MLflow
use_mlflow = False
mlflow_experiment_name = "stage2_model_training"

feature_importance_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

skipped_targets_csv = models_dir / "skipped_targets_stage2.csv"
cv_results_csv = models_dir / "model_cv_results_parallel.csv"
cv_results_json = models_dir / "model_cv_results_parallel.json"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage2")


def get_target_logger(y_col: str) -> logging.Logger:
    """Create a per-target logger writing to logs/y_<target>_stage2.log."""
    logger = logging.getLogger(f"stage2.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "_stage2_file", False)
        for handler in logger.handlers
    )
    if not exists:
        file_handler = logging.FileHandler(
            log_dir / f"{y_col}_stage2.log",
            mode="w",
            encoding="utf-8",
        )
        file_handler._stage2_file = True  # type: ignore[attr-defined]
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


def load_feature_importances_for_target(
    y_col: str,
    task_mode: str,
    logger: logging.Logger,
) -> Optional[List[str]]:
    """Load Stage-1 importance file for y_col and select usable features."""
    path = feature_importance_dir / f"feature_importances_{y_col}.csv"
    if not path.exists():
        logger.warning("No feature_importances file for %s at %s", y_col, path)
        return None

    df_imp = pd.read_csv(path)
    if "feature_name" not in df_imp.columns:
        logger.warning(
            "feature_importances_%s.csv has no 'feature_name' column; skipping.",
            y_col,
        )
        return None

    if task_mode == "regression":
        if "RF_REG" not in df_imp.columns:
            logger.warning(
                "feature_importances_%s.csv has no 'RF_REG'; skipping.",
                y_col,
            )
            return None
        mask = df_imp["RF_REG"] > 0
    else:
        required = ["RF", "LGBM", "CB", "XGB", "HGB"]
        missing = [c for c in required if c not in df_imp.columns]
        if missing:
            logger.warning(
                "feature_importances_%s.csv missing columns %s; skipping.",
                y_col,
                missing,
            )
            return None
        mask = (
            (df_imp["RF"] > 0)
            & (df_imp["LGBM"] > 0)
            & (df_imp["CB"] > 0)
            & (df_imp["XGB"] > 0)
            & (df_imp["HGB"] > 0)
        )

    selected = df_imp.loc[mask, "feature_name"].dropna().unique().tolist()
    if not selected:
        logger.warning(
            "No features selected for %s after importance filter; skipping.",
            y_col,
        )
        return None

    logger.info(
        "Selected %d features for %s from Stage-1 importances.",
        len(selected),
        y_col,
    )
    return selected


def prepare_views_classification(
    features: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare numeric and CatBoost views for classification."""
    numeric = features.copy()
    cat_cols = numeric.select_dtypes(include=["object"]).columns.tolist()

    if use_catboost_encoder and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
        numeric = encoder.fit_transform(numeric, y)
    elif cat_cols:
        numeric = numeric.drop(columns=cat_cols)

    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    cb_view = features.copy()
    cb_cat_cols = cb_view.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        cb_view[col] = cb_view[col].astype("string").fillna(cat_fill_value)

    return numeric, cb_view


def prepare_view_regression(
    features: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Prepare numeric CatBoost-encoded view for regression."""
    numeric = features.copy()
    cat_cols = numeric.select_dtypes(include=["object"]).columns.tolist()

    if use_catboost_encoder and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=random_state)
        numeric = encoder.fit_transform(numeric, y)
    elif cat_cols:
        numeric = numeric.drop(columns=cat_cols)

    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    return numeric


def choose_stratified_cv(y: pd.Series, logger: logging.Logger) -> Optional[StratifiedKFold]:
    """Choose dynamic StratifiedKFold for classification based on class counts."""
    counts = y.value_counts()
    min_count = int(counts.min())
    n_splits = min(max_n_splits_classification, min_count)

    logger.info(
        "Class distribution: %s; chosen n_splits=%d",
        counts.to_dict(),
        n_splits,
    )

    if n_splits < 2:
        logger.warning(
            "Cannot build StratifiedKFold: min_count=%d < 2; skipping target.",
            min_count,
        )
        return None

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def choose_kfold_regression(
    n_samples: int,
    logger: logging.Logger,
) -> Optional[KFold]:
    """Choose dynamic KFold for regression based on sample size."""
    n_splits = min(max_n_splits_regression, n_samples)
    logger.info("Regression KFold: n_samples=%d, n_splits=%d", n_samples, n_splits)

    if n_splits < 2:
        logger.warning(
            "Cannot build KFold: n_splits=%d < 2; skipping target.",
            n_splits,
        )
        return None

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------------------------

def build_classification_models(is_binary: bool) -> Dict[str, object]:
    """Build classification models (RF, LGBM, XGB, HGB, CB)."""
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    models: Dict[str, object] = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            objective=lgbm_obj,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
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
            random_state=random_state,
            n_jobs=-1,
        ),
        "HGB": HistGradientBoostingClassifier(
            max_depth=None,
            random_state=random_state,
        ),
        "CB": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function=cb_loss,
            random_state=random_state,
            verbose=False,
        ),
    }
    return models


def build_regression_models() -> Dict[str, object]:
    """Build regression models according to REGRESSION_MODELS."""
    models: Dict[str, object] = {}

    if "RF_REG" in REGRESSION_MODELS:
        models["RF_REG"] = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    if "LGBM_REG" in REGRESSION_MODELS:
        models["LGBM_REG"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    if "XGB_REG" in REGRESSION_MODELS:
        models["XGB_REG"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if "HGB_REG" in REGRESSION_MODELS:
        models["HGB_REG"] = HistGradientBoostingRegressor(
            max_depth=None,
            random_state=random_state,
        )
    if "CB_REG" in REGRESSION_MODELS:
        models["CB_REG"] = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_state=random_state,
            verbose=False,
        )

    return models


# ---------------------------------------------------------------------------
# CV EVALUATION HELPERS
# ---------------------------------------------------------------------------

def eval_classification_model_cv(
    name: str,
    model_proto: object,
    X_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Cross-validate one classification model."""
    f1_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    accuracy_scores: List[float] = []
    auc_scores: List[float] = []

    is_binary = y.nunique() == 2

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_num, y), start=1):
        X_train_num = X_num.iloc[train_idx]
        X_val_num = X_num.iloc[val_idx]
        X_train_cb = X_cb.iloc[train_idx]
        X_val_cb = X_cb.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if name == "CB":
            cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
            cat_indices = [X_cb.columns.get_loc(col) for col in cb_cat_cols]
            model = CatBoostClassifier(**model_proto.get_params())
            model.fit(
                X_train_cb,
                y_train,
                cat_features=cat_indices if cat_indices else None,
                verbose=False,
            )
            y_pred = model.predict(X_val_cb)
            proba = model.predict_proba(X_val_cb)
        else:
            if name == "HGB":
                model = HistGradientBoostingClassifier(**model_proto.get_params())
                model.fit(X_train_num, y_train)
                proba = model.predict_proba(X_val_num)
                y_pred = np.argmax(proba, axis=1)
            elif name == "LGBM":
                model = LGBMClassifier(**model_proto.get_params())
                model.fit(X_train_num, y_train)
                proba = model.predict_proba(X_val_num)
                y_pred = np.argmax(proba, axis=1)
            elif name == "RF":
                model = RandomForestClassifier(**model_proto.get_params())
                model.fit(X_train_num, y_train)
                proba = model.predict_proba(X_val_num)
                y_pred = model.predict(X_val_num)
            elif name == "XGB":
                model = XGBClassifier(**model_proto.get_params())
                model.fit(X_train_num, y_train)
                proba = model.predict_proba(X_val_num)
                y_pred = np.argmax(proba, axis=1)
            else:
                raise ValueError(f"Unknown classification model name: {name}")

        f1_scores.append(f1_score(y_val, y_pred, average="macro"))
        precision_scores.append(
            precision_score(y_val, y_pred, average="macro", zero_division=0),
        )
        recall_scores.append(
            recall_score(y_val, y_pred, average="macro", zero_division=0),
        )
        accuracy_scores.append(accuracy_score(y_val, y_pred))

        try:
            if is_binary:
                auc_scores.append(
                    roc_auc_score(y_val, proba[:, 1]),
                )
            else:
                auc_scores.append(
                    roc_auc_score(y_val, proba, multi_class="ovr"),
                )
        except Exception as exc:  # pragma: no cover - rare case
            logger.warning(
                "AUC computation failed for model %s, fold %d: %s",
                name,
                fold,
                exc,
            )

    return {
        "f1_macro": float(np.mean(f1_scores)),
        "precision_macro": float(np.mean(precision_scores)),
        "recall_macro": float(np.mean(recall_scores)),
        "accuracy": float(np.mean(accuracy_scores)),
        "auc": float(np.mean(auc_scores)) if auc_scores else float("nan"),
    }


def eval_regression_model_cv(
    name: str,
    model_proto: object,
    X_num: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> Dict[str, float]:
    """Cross-validate one regression model."""
    rmse_scores: List[float] = []
    mae_scores: List[float] = []
    r2_scores: List[float] = []

    for train_idx, val_idx in cv.split(X_num):
        X_train = X_num.iloc[train_idx]
        X_val = X_num.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if name == "RF_REG":
            model = RandomForestRegressor(**model_proto.get_params())
        elif name == "LGBM_REG":
            model = LGBMRegressor(**model_proto.get_params())
        elif name == "XGB_REG":
            model = XGBRegressor(**model_proto.get_params())
        elif name == "HGB_REG":
            model = HistGradientBoostingRegressor(**model_proto.get_params())
        elif name == "CB_REG":
            model = CatBoostRegressor(**model_proto.get_params())
        else:
            raise ValueError(f"Unknown regression model name: {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_scores.append(mean_squared_error(y_val, y_pred, squared=False))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        r2_scores.append(r2_score(y_val, y_pred))

    return {
        "rmse": float(np.mean(rmse_scores)),
        "mae": float(np.mean(mae_scores)),
        "r2": float(np.mean(r2_scores)),
    }


# ---------------------------------------------------------------------------
# PER-TARGET PROCESSING
# ---------------------------------------------------------------------------

def process_target(
    y_col: str,
    df: pd.DataFrame,
    id_cols: List[str],
    base_ft_cols: List[str],
) -> Dict[str, object]:
    """Process one target for Stage-2 training."""
    logger = get_target_logger(y_col)
    logger.info("=== Stage-2 training for %s (mode=%s) ===", y_col, TASK_MODE)

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

        nunique = y_raw.nunique(dropna=True)
        if nunique <= REGRESSION_MIN_UNIQUE:
            logger.warning(
                (
                    "Skipping %s: only %d unique numeric values (<= %d); "
                    "treated as categorical, not regression."
                ),
                y_col,
                nunique,
                REGRESSION_MIN_UNIQUE,
            )
            return {
                "target": y_col,
                "skipped": True,
                "reason": "too_few_unique_for_regression",
                "n_rows": int(n_rows),
            }

        y = pd.to_numeric(y_raw, errors="coerce")
        valid_mask = y.notna()
        df_target = df_target[valid_mask]
        y = y[valid_mask]
        n_rows = df_target.shape[0]

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

        logger.info(
            "Regression target %s accepted: %d unique values, %d usable rows.",
            y_col,
            nunique,
            n_rows,
        )

        cv = choose_kfold_regression(n_rows, logger)
        if cv is None:
            return {
                "target": y_col,
                "skipped": True,
                "reason": "cv_failed",
                "n_rows": int(n_rows),
            }

    else:
        # classification
        y = y_raw.astype(str)
        counts = y.value_counts()
        n_classes = counts.shape[0]
        min_class = int(counts.min())

        logger.info(
            "Classification target %s: n_classes=%d, min_class=%d, counts=%s",
            y_col,
            n_classes,
            min_class,
            counts.to_dict(),
        )

        if n_classes < 2:
            logger.warning("Skipping %s: only one class.", y_col)
            return {
                "target": y_col,
                "skipped": True,
                "reason": "single_class",
                "n_rows": int(n_rows),
            }

        if min_class < min_class_count_for_training:
            logger.warning(
                "Skipping %s: min_class=%d (< %d).",
                y_col,
                min_class,
                min_class_count_for_training,
            )
            return {
                "target": y_col,
                "skipped": True,
                "reason": "rare_class",
                "n_rows": int(n_rows),
            }

        y_enc = pd.Series(
            pd.factorize(y, sort=True)[0],
            index=y.index,
            dtype="int64",
        )
        y = y_enc

        cv = choose_stratified_cv(y, logger)
        if cv is None:
            return {
                "target": y_col,
                "skipped": True,
                "reason": "cv_failed",
                "n_rows": int(n_rows),
            }

    selected_features = load_feature_importances_for_target(
        y_col=y_col,
        task_mode=TASK_MODE,
        logger=logger,
    )
    if not selected_features:
        return {
            "target": y_col,
            "skipped": True,
            "reason": "no_features_selected",
            "n_rows": int(n_rows),
        }

    all_features = [c for c in base_ft_cols if c in selected_features]
    if not all_features:
        logger.warning(
            "None of the selected features for %s are in base_ft_cols; skipping.",
            y_col,
        )
        return {
            "target": y_col,
            "skipped": True,
            "reason": "selected_features_not_in_df",
            "n_rows": int(n_rows),
        }

    X = df_target[all_features].copy()

    run = None
    if use_mlflow and mlflow_available and mlflow is not None:
        run = mlflow.start_run(run_name=f"stage2_{y_col}", nested=False)
        mlflow.log_param("target", y_col)
        mlflow.log_param("task_mode", TASK_MODE)
        mlflow.log_param("n_rows", int(n_rows))
        mlflow.log_param("n_features", len(all_features))
        mlflow.log_param("selected_features", ",".join(all_features))

    if TASK_MODE == "regression":
        # -------- Regression branch --------
        X_num = prepare_view_regression(X, y)
        models = build_regression_models()
        if not models:
            logger.warning(
                "No regression models enabled in REGRESSION_MODELS; skipping %s.",
                y_col,
            )
            return {
                "target": y_col,
                "skipped": True,
                "reason": "no_regression_models",
                "n_rows": int(n_rows),
            }

        model_metrics: Dict[str, Dict[str, float]] = {}
        for name, proto in models.items():
            logger.info("CV for regression model %s on %s", name, y_col)
            metrics = eval_regression_model_cv(
                name=name,
                model_proto=proto,
                X_num=X_num,
                y=y,
                cv=cv,
            )
            model_metrics[name] = metrics
            logger.info("CV metrics for %s on %s: %s", name, y_col, metrics)

        # Best by lowest RMSE
        best_name = min(
            model_metrics.items(),
            key=lambda kv: kv[1]["rmse"],
        )[0]
        logger.info("Best regression model for %s is %s", y_col, best_name)

        fitted_paths: Dict[str, str] = {}

        for name, proto in models.items():
            if name == "RF_REG":
                model = RandomForestRegressor(**proto.get_params())
            elif name == "LGBM_REG":
                model = LGBMRegressor(**proto.get_params())
            elif name == "XGB_REG":
                model = XGBRegressor(**proto.get_params())
            elif name == "HGB_REG":
                model = HistGradientBoostingRegressor(**proto.get_params())
            elif name == "CB_REG":
                model = CatBoostRegressor(**proto.get_params())
            else:
                continue

            model.fit(X_num, y)
            out_path = models_dir / f"{y_col}_{name}.joblib"
            dump(model, out_path)
            fitted_paths[name] = str(out_path)
            logger.info("Saved regression model %s for %s to %s", name, y_col, out_path)

        # Best model alias
        best_src_path = fitted_paths[best_name]
        best_model = load(best_src_path)
        best_path = models_dir / f"{y_col}_best.joblib"
        dump(best_model, best_path)
        logger.info("Saved best regression model alias for %s to %s", y_col, best_path)

        if use_mlflow and mlflow_available and run is not None and mlflow is not None:
            best_metrics = model_metrics[best_name]
            mlflow.log_metrics(best_metrics)
            mlflow.log_param("best_model", best_name)
            mlflow.end_run()

        flat_metrics: Dict[str, float] = {}
        for name, metrics in model_metrics.items():
            for key, val in metrics.items():
                flat_metrics[f"{name}_{key}"] = val

        return {
            "target": y_col,
            "skipped": False,
            "reason": "",
            "n_rows": int(n_rows),
            "task_mode": TASK_MODE,
            "best_model": best_name,
            **flat_metrics,
        }

    # -------- Classification branch --------
    X_num, X_cb = prepare_views_classification(X, y)
    models = build_classification_models(is_binary=y.nunique() == 2)

    model_metrics_cls: Dict[str, Dict[str, float]] = {}
    for name, proto in models.items():
        logger.info("CV for classification model %s on %s", name, y_col)
        metrics = eval_classification_model_cv(
            name=name,
            model_proto=proto,
            X_num=X_num,
            X_cb=X_cb,
            y=y,
            cv=cv,
            logger=logger,
        )
        model_metrics_cls[name] = metrics
        logger.info("CV metrics for %s on %s: %s", name, y_col, metrics)

    best_name = max(
        model_metrics_cls.items(),
        key=lambda kv: kv[1]["f1_macro"],
    )[0]
    logger.info("Best classification model for %s is %s", y_col, best_name)

    fitted_paths_cls: Dict[str, str] = {}

    for name, proto in models.items():
        if name == "CB":
            cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
            cat_indices = [X_cb.columns.get_loc(col) for col in cb_cat_cols]
            model = CatBoostClassifier(**proto.get_params())
            model.fit(
                X_cb,
                y,
                cat_features=cat_indices if cat_indices else None,
                verbose=False,
            )
        elif name == "HGB":
            model = HistGradientBoostingClassifier(**proto.get_params())
            model.fit(X_num, y)
        elif name == "LGBM":
            model = LGBMClassifier(**proto.get_params())
            model.fit(X_num, y)
        elif name == "RF":
            model = RandomForestClassifier(**proto.get_params())
            model.fit(X_num, y)
        elif name == "XGB":
            model = XGBClassifier(**proto.get_params())
            model.fit(X_num, y)
        else:
            continue

        out_path = models_dir / f"{y_col}_{name}.joblib"
        dump(model, out_path)
        fitted_paths_cls[name] = str(out_path)
        logger.info("Saved classification model %s for %s to %s", name, y_col, out_path)

    best_src_path_cls = fitted_paths_cls[best_name]
    best_model_cls = load(best_src_path_cls)
    best_path_cls = models_dir / f"{y_col}_best.joblib"
    dump(best_model_cls, best_path_cls)
    logger.info("Saved best classification model alias for %s to %s", y_col, best_path_cls)

    if use_mlflow and mlflow_available and run is not None and mlflow is not None:
        best_metrics_cls = model_metrics_cls[best_name]
        mlflow.log_metrics(best_metrics_cls)
        mlflow.log_param("best_model", best_name)
        mlflow.end_run()

    flat_metrics_cls: Dict[str, float] = {}
    for name, metrics in model_metrics_cls.items():
        for key, val in metrics.items():
            flat_metrics_cls[f"{name}_{key}"] = val

    return {
        "target": y_col,
        "skipped": False,
        "reason": "",
        "n_rows": int(n_rows),
        "task_mode": TASK_MODE,
        "best_model": best_name,
        **flat_metrics_cls,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Stage-2 training over all y_* targets."""
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

    if use_mlflow and mlflow_available and mlflow is not None:
        mlflow.set_experiment(mlflow_experiment_name)

    log.info(
        "Starting Stage-2 over %d targets with n_jobs_targets=%d (mode=%s).",
        len(y_cols),
        n_jobs_targets,
        TASK_MODE,
    )

    results = Parallel(n_jobs=n_jobs_targets)(
        delayed(process_target)(y_col, df, id_cols, ft_cols) for y_col in y_cols
    )

    skipped = [r for r in results if r.get("skipped")]
    processed = [r for r in results if not r.get("skipped")]

    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_targets_csv, index=False)
        log.info("Saved skipped targets summary to %s", skipped_targets_csv)

    if processed:
        df_cv = pd.DataFrame(processed)
        df_cv.to_csv(cv_results_csv, index=False)
        with cv_results_json.open("w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        log.info("Saved CV results to %s and %s", cv_results_csv, cv_results_json)
    else:
        log.warning("No targets were successfully processed in Stage-2.")

    log.info(
        "Stage-2 completed: %d processed, %d skipped.",
        len(processed),
        len(skipped),
    )


if __name__ == "__main__":
    main()
