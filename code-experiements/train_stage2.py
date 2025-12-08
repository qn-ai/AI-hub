#!/usr/bin/env python3
"""
Stage-2: Per-target model training with dynamic CV folds and
saving one trained model per algorithm (RF, LGBM, XGB, HGB, CB).

This stage:

- Uses feature_importances_<y>.csv from Stage-1 to choose features per target.
- For each y_*:
    * Filters non-missing rows.
    * Checks class distribution and chooses n_splits = min(MAX_N_SPLITS,
      min_class_count).
    * Skips targets with too few rows or degenerate classes.
    * Builds two feature views:
        - Numeric (CatBoostEncoder) for RF / LGBM / XGB / HGB.
        - Raw string categorical view for CatBoost.
    * Cross-validates all 5 models and computes metrics:
        - F1, Precision, Recall, Accuracy, AUC.
    * Selects the best model by F1.
    * Refits ALL 5 models on the full target data.
    * Saves:
        - trained_models/y_<target>_RF.joblib
        - trained_models/y_<target>_LGBM.joblib
        - trained_models/y_<target>_XGB.joblib
        - trained_models/y_<target>_HGB.joblib
        - trained_models/y_<target>_CB.joblib
        - trained_models/y_<target>_best.joblib (alias to best model).

Outputs:

- model_cv_results_parallel.csv
- model_cv_results_parallel.json
- skipped_targets_stage2.csv
- logs/y_<target>_stage2.log

MLflow is optional and disabled by default.
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

# Optional MLflow
try:
    import mlflow  # type: ignore[import]

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - MLflow optional
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

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
MAX_N_SPLITS = 5

CPU_COUNT = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

# MLflow toggle
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "Stage2_ModelTraining"

TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        isinstance(h, logging.FileHandler) and getattr(h, "_stage2_file", False)
        for h in logger.handlers
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
    """Detect id_, ft_, y_ columns by prefix.

    Args:
        df: Input dataframe.

    Returns:
        Tuple of lists (id_cols, ft_cols, y_cols).
    """
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_selected_features(y_col: str, logger: logging.Logger) -> List[str]:
    """Load selected features from feature_importances_<y>.csv.

    Only keep features where RF, LGBM, CB, XGB, HGB > 0.

    Args:
        y_col: Target column name.
        logger: Logger for messages.

    Returns:
        List of feature names to use for this target.
    """
    path = FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
    if not path.exists():
        logger.warning("Feature importance file not found: %s", path)
        return []

    df_imp = pd.read_csv(path)
    if "feature_name" not in df_imp.columns:
        # Fallback if first column is feature names.
        df_imp = df_imp.rename(columns={df_imp.columns[0]: "feature_name"})

    required = ["RF", "LGBM", "CB", "XGB", "HGB"]
    missing = [c for c in required if c not in df_imp.columns]
    if missing:
        logger.warning("Importance file %s missing columns %s", path, missing)
        return []

    mask = (df_imp[required] > 0).all(axis=1)
    selected = df_imp.loc[mask, "feature_name"].astype(str).tolist()
    logger.info("Selected %d features for %s", len(selected), y_col)
    return selected


def prepare_target_and_cv(
    y_raw: pd.Series,
    logger: logging.Logger,
) -> Tuple[Optional[pd.Series], Optional[LabelEncoder], Optional[int]]:
    """Encode target labels and decide on number of CV folds.

    Args:
        y_raw: Raw target series with non-missing values.
        logger: Logger for this target.

    Returns:
        Tuple (y_encoded, label_encoder, n_splits).
        If y_encoded is None, target should be skipped.
    """
    y_str = y_raw.astype(str)
    counts = y_str.value_counts()
    n_classes = counts.shape[0]
    min_count = int(counts.min())

    logger.info("Class distribution: %s", counts.to_dict())

    if min_count < 2:
        logger.warning(
            "Skipping target: min class count = %d < 2 (n_classes=%d).",
            min_count,
            n_classes,
        )
        return None, None, None

    n_splits = min(MAX_N_SPLITS, min_count)
    if n_splits < 2:
        logger.warning("Skipping target: computed n_splits=%d < 2.", n_splits)
        return None, None, None

    encoder = LabelEncoder()
    y_enc = pd.Series(
        encoder.fit_transform(y_str),
        index=y_raw.index,
        dtype="int64",
    )

    logger.info(
        "Using n_splits=%d (n_classes=%d, min_class_count=%d, labels=%s).",
        n_splits,
        n_classes,
        min_count,
        np.unique(y_enc).tolist(),
    )
    return y_enc, encoder, n_splits


def prepare_catboost_view(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare CatBoost view: string categoricals, NaNs -> CAT_FILL_VALUE.

    Args:
        X: Raw feature dataframe.

    Returns:
        Tuple (X_cb, cat_cols).
    """
    X_cb = X.copy()
    cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype("string").fillna(CAT_FILL_VALUE)
    return X_cb, cat_cols


def prepare_numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare numeric view with CatBoostEncoder for tree-based models.

    Args:
        X: Raw feature dataframe.

    Returns:
        Numeric dataframe with NaNs preserved.
    """
    X_num = X.copy()
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        dummy_y = np.zeros(len(X_num))
        X_num = encoder.fit_transform(X_num, dummy_y)
    elif cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    return X_num


def build_models(is_binary: bool) -> Dict[str, object]:
    """Build model prototypes for this target.

    Args:
        is_binary: True if target has 2 classes.

    Returns:
        Dictionary mapping model name to model prototype.
    """
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


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    is_binary: bool,
) -> Dict[str, float]:
    """Compute evaluation metrics for one fold.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (can be None).
        is_binary: Whether the problem is binary.

    Returns:
        Dictionary with accuracy, f1, precision, recall, auc.
    """
    average = "binary" if is_binary else "macro"
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    if y_proba is None:
        metrics["auc"] = float("nan")
        return metrics

    try:
        if is_binary:
            if y_proba.ndim == 1:
                metrics["auc"] = roc_auc_score(y_true, y_proba)
            else:
                metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["auc"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
            )
    except Exception:
        metrics["auc"] = float("nan")

    return metrics


def compute_proba_for_fold(
    model: object,
    X_val: pd.DataFrame,
    is_binary: bool,
) -> Optional[np.ndarray]:
    """Compute probabilities for validation set if supported by model."""
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X_val)
    if proba.ndim == 1:
        return proba

    if is_binary and proba.shape[1] == 2:
        return proba[:, 1]
    return proba


def cross_validate_target(
    X: pd.DataFrame,
    y: pd.Series,
    X_cb: pd.DataFrame,
    cat_cols_cb: List[str],
    n_splits: int,
    logger: logging.Logger,
) -> Tuple[Optional[str], Dict[str, Dict[str, float]], Dict[str, object]]:
    """Run CV over all models for one target.

    Args:
        X: Numeric feature view.
        y: Encoded labels.
        X_cb: CatBoost feature view.
        cat_cols_cb: CatBoost categorical columns.
        n_splits: Number of CV folds.
        logger: Logger for this target.

    Returns:
        best_model_name, metrics_per_model, trained_models_on_full_data.
    """
    is_binary = y.nunique() == 2
    models_proto = build_models(is_binary=is_binary)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols_cb]
    model_results: Dict[str, Dict[str, float]] = {}
    best_model_name: Optional[str] = None
    best_f1 = -np.inf

    for name, proto in models_proto.items():
        logger.info("Cross-validating model: %s", name)
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            logger.info("  Fold %d/%d for %s", fold_idx, n_splits, name)
            if name == "CB":
                X_train = X_cb.iloc[train_idx]
                X_val = X_cb.iloc[val_idx]
            else:
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = proto.__class__(**proto.get_params())
            if name == "CB":
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_indices if cat_indices else None,
                )
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = compute_proba_for_fold(model, X_val, is_binary)
            metrics = compute_fold_metrics(
                y_true=y_val.to_numpy(),
                y_pred=y_pred,
                y_proba=y_proba,
                is_binary=is_binary,
            )
            fold_metrics.append(metrics)

        agg = {
            metric_name: float(np.nanmean([m[metric_name] for m in fold_metrics]))
            for metric_name in fold_metrics[0].keys()
        }
        agg["n_splits"] = float(n_splits)
        model_results[name] = agg
        logger.info("Aggregated CV metrics for %s: %s", name, agg)

        if agg["f1"] > best_f1:
            best_f1 = agg["f1"]
            best_model_name = name

    if best_model_name is None:
        logger.warning("No valid model found (all F1 were -inf).")
        return None, model_results, {}

    logger.info("Best model: %s (F1=%.4f)", best_model_name, best_f1)

    # Refit all models on full data.
    trained_models: Dict[str, object] = {}
    for name, proto in models_proto.items():
        logger.info("Refitting full model: %s", name)
        model = proto.__class__(**proto.get_params())
        if name == "CB":
            model.fit(
                X_cb,
                y,
                cat_features=cat_indices if cat_indices else None,
            )
        else:
            X_num_full = prepare_numeric_view(X)
            model.fit(X_num_full, y)
        trained_models[name] = model

    return best_model_name, model_results, trained_models


# ---------------------------------------------------------------------------
# PER-TARGET PIPELINE
# ---------------------------------------------------------------------------

def process_target(y_col: str, df: pd.DataFrame) -> Dict[str, object]:
    """Process training for a single target column.

    Args:
        y_col: Target column name.
        df: Full dataframe.

    Returns:
        Record describing processing result for this target.
    """
    logger = get_target_logger(y_col)
    logger.info("=== Stage-2 training for %s ===", y_col)

    df_t = df[df[y_col].notna()].copy()
    n_rows = df_t.shape[0]
    if n_rows < 2:
        logger.warning("Skipping %s: only %d labelled rows (< 2).", y_col, n_rows)
        return {"target": y_col, "skipped": True, "reason": "too_few_rows"}

    selected_features = load_selected_features(y_col, logger)
    if not selected_features:
        logger.warning("Skipping %s: no selected features.", y_col)
        return {"target": y_col, "skipped": True, "reason": "no_selected_features"}

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        logger.warning(
            "Skipping %s: missing features in data: %s",
            y_col,
            missing,
        )
        return {"target": y_col, "skipped": True, "reason": "missing_features"}

    X = df_t[selected_features].copy()
    y_raw = df_t[y_col]

    y_enc, encoder, n_splits = prepare_target_and_cv(y_raw, logger)
    if y_enc is None or encoder is None or n_splits is None:
        return {"target": y_col, "skipped": True, "reason": "bad_class_distribution"}

    X_cb, cat_cols_cb = prepare_catboost_view(X)
    X_num = prepare_numeric_view(X)

    # Optional MLflow logging for this target.
    run = None
    if USE_MLFLOW and MLFLOW_AVAILABLE:
        if mlflow is not None:
            run = mlflow.start_run(run_name=f"stage2_{y_col}", nested=False)
            mlflow.log_param("target", y_col)
            mlflow.log_param("n_rows", int(n_rows))
            mlflow.log_param("n_features", int(X_num.shape[1]))
            mlflow.log_param("n_splits", int(n_splits))

    best_model_name, metrics_per_model, trained_models = cross_validate_target(
        X=X_num,
        y=y_enc,
        X_cb=X_cb,
        cat_cols_cb=cat_cols_cb,
        n_splits=n_splits,
        logger=logger,
    )

    if not trained_models or best_model_name is None:
        logger.warning("No trained models produced for %s.", y_col)
        if USE_MLFLOW and MLFLOW_AVAILABLE and run is not None and mlflow is not None:
            mlflow.set_tag("status", "no_valid_model")
            mlflow.end_run()
        return {"target": y_col, "skipped": True, "reason": "no_valid_model"}

    # Save all model files.
    for model_name, est in trained_models.items():
        model_path = TRAINED_MODELS_DIR / f"{y_col}_{model_name}.joblib"
        joblib.dump(est, model_path)
        logger.info("Saved %s model for %s to %s", model_name, y_col, model_path)

    # Save alias for best model.
    best_model = trained_models[best_model_name]
    best_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    joblib.dump(best_model, best_path)
    logger.info("Saved BEST model (%s) for %s to %s", best_model_name, y_col, best_path)

    best_metrics = metrics_per_model[best_model_name]
    record: Dict[str, object] = {
        "target": y_col,
        "skipped": False,
        "reason": "",
        "best_model": best_model_name,
        "n_splits": best_metrics.get("n_splits", float(n_splits)),
        "f1": best_metrics.get("f1", float("nan")),
        "precision": best_metrics.get("precision", float("nan")),
        "recall": best_metrics.get("recall", float("nan")),
        "accuracy": best_metrics.get("accuracy", float("nan")),
        "auc": best_metrics.get("auc", float("nan")),
    }
    logger.info("Final record for %s: %s", y_col, record)

    if USE_MLFLOW and MLFLOW_AVAILABLE and run is not None and mlflow is not None:
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("f1", float(record["f1"]))
        mlflow.log_metric("precision", float(record["precision"]))
        mlflow.log_metric("recall", float(record["recall"]))
        mlflow.log_metric("accuracy", float(record["accuracy"]))
        if not np.isnan(record["auc"]):  # type: ignore[arg-type]
            mlflow.log_metric("auc", float(record["auc"]))
        mlflow.set_tag("status", "success")
        mlflow.end_run()

    return record


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Stage-2 model training over all y_* targets."""
    LOG.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    id_cols, ft_cols, y_cols = detect_columns(df)
    LOG.info(
        "Detected %d id_, %d ft_, %d y_ columns.",
        len(id_cols),
        len(ft_cols),
        len(y_cols),
    )

    if not y_cols:
        LOG.error("No y_ columns found; aborting Stage-2.")
        return

    if USE_MLFLOW and MLFLOW_AVAILABLE and mlflow is not None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    LOG.info(
        "Starting Stage-2 for %d targets with N_JOBS_TARGETS=%d.",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(process_target)(y_col, df) for y_col in y_cols
    )

    processed = [r for r in results if not r.get("skipped", False)]
    skipped = [r for r in results if r.get("skipped", False)]

    if processed:
        df_processed = pd.DataFrame(processed)
        df_processed.to_csv(RESULTS_CSV, index=False)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        LOG.info(
            "Saved Stage-2 metrics to %s and %s.",
            RESULTS_CSV,
            RESULTS_JSON,
        )
    else:
        LOG.warning("No targets successfully trained in Stage-2.")

    if skipped:
        df_skipped = pd.DataFrame(skipped)
        df_skipped.to_csv(SKIPPED_CSV, index=False)
        LOG.info("Saved skipped targets to %s.", SKIPPED_CSV)
    else:
        LOG.info("No targets skipped in Stage-2.")

    LOG.info(
        "Stage-2 completed: %d processed, %d skipped.",
        len(processed),
        len(skipped),
    )


if __name__ == "__main__":
    main()
