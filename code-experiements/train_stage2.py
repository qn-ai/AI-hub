#!/usr/bin/env python3
"""
Stage-2: Per-target classification model training with dynamic CV folds.

This script:

- Uses feature_importances_<y>.csv from Stage-1 to choose features per target.
- Classification only (no regression).
- For each y_* target:
    * Filters non-missing rows.
    * Skips targets with fewer than MIN_SAMPLES_PER_TARGET rows.
    * Skips targets with only one class or smallest class too small.
    * Loads feature_importances_<y>.csv and selects features where:
        RF > 0 & LGBM > 0 & CB > 0 & XGB > 0 & HGB > 0.
    * Builds two feature views:
        - Numeric (CatBoostEncoder) for RF / LGBM / XGB / HGB.
        - Raw string categorical view for CatBoost.
    * Chooses dynamic n_splits as:
        n_splits = min(MAX_N_SPLITS_CLASSIFICATION, min_class_count)
        (must be >= 2).
    * Cross-validates all ENABLED_MODELS and computes metrics:
        F1 (macro), Precision (macro), Recall (macro),
        Accuracy, AUC (binary or multiclass).
    * Selects best model by F1.
    * Refits all ENABLED_MODELS on full data.
    * Saves:
        trained_models/y_<target>_<MODEL>.joblib
        trained_models/y_<target>_best.joblib

Outputs:

- trained_models/model_cv_results_parallel.csv
- trained_models/model_cv_results_parallel.json
- trained_models/skipped_targets_stage2.csv
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
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, dump, load
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Optional MLflow
try:
    import mlflow

    mlflow_available = True
except Exception:  # pragma: no cover - optional
    mlflow = None  # type: ignore[assignment]
    mlflow_available = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_PATH = "input_data.csv"
FEATURE_IMPORTANCE_DIR = Path("feature_importances")
MODELS_DIR = Path("trained_models")
LOG_DIR = Path("logs")

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"

RANDOM_STATE = 42

MIN_SAMPLES_PER_TARGET = 200
MIN_CLASS_COUNT_FOR_TRAINING = 2
MAX_N_SPLITS_CLASSIFICATION = 5

CPU_COUNT = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

# ✔ Select which models to run for classification
# Any subset of {"RF", "LGBM", "XGB", "HGB", "CB"}.
ENABLED_MODELS: List[str] = ["RF", "LGBM", "XGB", "HGB", "CB"]

# MLflow (optional)
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "stage2_model_training"

FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SKIPPED_TARGETS_CSV = MODELS_DIR / "skipped_targets_stage2.csv"
CV_RESULTS_CSV = MODELS_DIR / "model_cv_results_parallel.csv"
CV_RESULTS_JSON = MODELS_DIR / "model_cv_results_parallel.json"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("stage2")


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
            LOG_DIR / f"{y_col}_stage2.log",
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
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_feature_importances_for_target(
    y_col: str,
    logger: logging.Logger,
) -> Optional[List[str]]:
    """Load Stage-1 importance file for y_col and select usable features."""
    path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
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

    if USE_CATBOOST_ENCODER and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        numeric = encoder.fit_transform(numeric, y)
    elif cat_cols:
        numeric = numeric.drop(columns=cat_cols)

    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    cb_view = features.copy()
    cb_cat_cols = cb_view.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        cb_view[col] = cb_view[col].astype("string").fillna(CAT_FILL_VALUE)

    return numeric, cb_view


def choose_stratified_cv(
    y: pd.Series,
    logger: logging.Logger,
) -> Optional[StratifiedKFold]:
    """Choose dynamic StratifiedKFold for classification based on class counts."""
    counts = y.value_counts()
    min_count = int(counts.min())
    n_splits = min(MAX_N_SPLITS_CLASSIFICATION, min_count)

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
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------------------------


def build_classification_models(is_binary: bool) -> Dict[str, object]:
    """Build enabled classification models."""
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    models: Dict[str, object] = {}

    if "RF" in ENABLED_MODELS:
        models["RF"] = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if "LGBM" in ENABLED_MODELS:
        models["LGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            objective=lgbm_obj,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if "XGB" in ENABLED_MODELS:
        models["XGB"] = XGBClassifier(
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
        )

    if "HGB" in ENABLED_MODELS:
        models["HGB"] = HistGradientBoostingClassifier(
            max_depth=None,
            random_state=RANDOM_STATE,
        )

    if "CB" in ENABLED_MODELS:
        models["CB"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function=cb_loss,
            random_state=RANDOM_STATE,
            verbose=False,
        )

    return models


# ---------------------------------------------------------------------------
# CV EVALUATION
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
                auc_scores.append(roc_auc_score(y_val, proba[:, 1]))
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


# ---------------------------------------------------------------------------
# PER-TARGET PROCESSING
# ---------------------------------------------------------------------------


def process_target(
    y_col: str,
    df: pd.DataFrame,
    ft_cols: List[str],
) -> Dict[str, object]:
    """Process one target for Stage-2 training (classification only)."""
    logger = get_target_logger(y_col)
    logger.info("=== Stage-2 training for %s ===", y_col)

    df_target = df[df[y_col].notna()].copy()
    n_rows = df_target.shape[0]
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

    y_raw = df_target[y_col].astype(str)
    counts = y_raw.value_counts()
    n_classes = counts.shape[0]
    min_class = int(counts.min())

    logger.info(
        "Target %s: n_classes=%d, min_class=%d, counts=%s",
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

    if min_class < MIN_CLASS_COUNT_FOR_TRAINING:
        logger.warning(
            "Skipping %s: min_class=%d (< %d).",
            y_col,
            min_class,
            MIN_CLASS_COUNT_FOR_TRAINING,
        )
        return {
            "target": y_col,
            "skipped": True,
            "reason": "rare_class",
            "n_rows": int(n_rows),
        }

    y = pd.Series(
        pd.factorize(y_raw, sort=True)[0],
        index=y_raw.index,
        dtype="int64",
    )

    cv = choose_stratified_cv(y, logger)
    if cv is None:
        return {
            "target": y_col,
            "skipped": True,
            "reason": "cv_failed",
            "n_rows": int(n_rows),
        }

    selected_features = load_feature_importances_for_target(y_col, logger)
    if not selected_features:
        return {
            "target": y_col,
            "skipped": True,
            "reason": "no_features_selected",
            "n_rows": int(n_rows),
        }

    all_features = [c for c in ft_cols if c in selected_features]
    if not all_features:
        logger.warning(
            "None of the selected features for %s are in ft_cols; skipping.",
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
    if USE_MLFLOW and mlflow_available and mlflow is not None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=f"stage2_{y_col}")
        mlflow.log_param("target", y_col)
        mlflow.log_param("n_rows", int(n_rows))
        mlflow.log_param("n_features", len(all_features))
        mlflow.log_param("enabled_models", ",".join(ENABLED_MODELS))

    X_num, X_cb = prepare_views_classification(X, y)
    models = build_classification_models(is_binary=y.nunique() == 2)
    if not models:
        logger.warning("No ENABLED_MODELS for %s; skipping.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "no_enabled_models",
            "n_rows": int(n_rows),
        }

    model_metrics: Dict[str, Dict[str, float]] = {}
    for name, proto in models.items():
        logger.info("CV for model %s on %s", name, y_col)
        metrics = eval_classification_model_cv(
            name=name,
            model_proto=proto,
            X_num=X_num,
            X_cb=X_cb,
            y=y,
            cv=cv,
            logger=logger,
        )
        model_metrics[name] = metrics
        logger.info("CV metrics for %s on %s: %s", name, y_col, metrics)

    best_name = max(
        model_metrics.items(),
        key=lambda kv: kv[1]["f1_macro"],
    )[0]
    logger.info("Best model for %s is %s", y_col, best_name)

    fitted_paths: Dict[str, str] = {}

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

        out_path = MODELS_DIR / f"{y_col}_{name}.joblib"
        dump(model, out_path)
        fitted_paths[name] = str(out_path)
        logger.info("Saved model %s for %s to %s", name, y_col, out_path)

    best_src_path = fitted_paths[best_name]
    best_model = load(best_src_path)
    best_path = MODELS_DIR / f"{y_col}_best.joblib"
    dump(best_model, best_path)
    logger.info("Saved best model alias for %s to %s", y_col, best_path)

    if USE_MLFLOW and mlflow_available and run is not None and mlflow is not None:
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
        "best_model": best_name,
        **flat_metrics,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage-2 training over all y_* targets."""
    LOG.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    _, ft_cols, y_cols = detect_columns(df)
    LOG.info(
        "Detected %d ft_, %d y_ columns.",
        len(ft_cols),
        len(y_cols),
    )

    if not ft_cols or not y_cols:
        LOG.error("No ft_ or y_ columns detected; aborting.")
        return

    LOG.info(
        "Starting Stage-2 over %d targets with n_jobs_targets=%d.",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(process_target)(y_col, df, ft_cols) for y_col in y_cols
    )

    skipped = [r for r in results if r.get("skipped")]
    processed = [r for r in results if not r.get("skipped")]

    if skipped:
        pd.DataFrame(skipped).to_csv(SKIPPED_TARGETS_CSV, index=False)
        LOG.info("Saved skipped targets summary to %s", SKIPPED_TARGETS_CSV)

    if processed:
        df_cv = pd.DataFrame(processed)
        df_cv.to_csv(CV_RESULTS_CSV, index=False)
        with CV_RESULTS_JSON.open("w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        LOG.info("Saved CV results to %s and %s", CV_RESULTS_CSV, CV_RESULTS_JSON)
    else:
        LOG.warning("No targets were successfully processed in Stage-2.")

    LOG.info(
        "Stage-2 completed: %d processed, %d skipped.",
        len(processed),
        len(skipped),
    )


if __name__ == "__main__":
    main()
