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
            - Added SimpleImputer (only use if have to!!) for models which can't
            numeric NAs
        - Raw string categorical view for CatBoost.
    * Cross-validates all 5 models and computes metrics:
        - For classification: F1, Precision, Recall, Accuracy, AUC.
        - For regression: MAE, MSE, RMSE, R2
    * Selects the best model by F1 or RMSE
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
"""  # noqa: D205

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from typing import Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier
from joblib import Parallel, delayed, dump, load
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,  # noqa: F401
    balanced_accuracy_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from quantile_forest import RandomForestQuantileRegressor

from assessmentestimation.helpers import return_master_data, fetch_master_data_fn

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

# Global task mode:
#   "classification" → all targets treated as classification
#   "regression"     → all targets treated as regression (numeric only)

# Define task mode based on model type
MODEL_TYPE = "predictassessment"  # Options: "predictassessment", "predictbudget", "assessmentbudget"

DEV_ROW_SUBSET = None

CLASS_MODEL_TYPES = ["predictassessment"]
REGRESSION_MODEL_TYPES = ["predictbudget", "assessmentbudget"]
MODEL_TYPES = CLASS_MODEL_TYPES + REGRESSION_MODEL_TYPES

# Set TASK_MODE correctly
if MODEL_TYPE in CLASS_MODEL_TYPES:
    TASK_MODE = "classification"
elif MODEL_TYPE in REGRESSION_MODEL_TYPES:
    TASK_MODE = "regression"
else:
    raise ValueError(
        f"Invalid model type: {MODEL_TYPE}. Defined types are: {REGRESSION_MODEL_TYPES + CLASS_MODEL_TYPES}"
    )

DATA_PATH = "input_data.csv"

FEATURE_IMPORTANCE_DIR = Path(f"ae_models_pipeline/{MODEL_TYPE}/feature_importances")
TRAINED_MODELS_DIR = Path(f"ae_models_pipeline/{MODEL_TYPE}/trained_models")
LOG_DIR = Path(f"ae_models_pipeline/{MODEL_TYPE}/logs")

RESULTS_CSV = f"{TASK_MODE}_models_pipeline/model_cv_results_parallel.csv"
RESULTS_JSON = f"{TASK_MODE}_models_pipeline/model_cv_results_parallel.json"
SKIPPED_CSV = f"{TASK_MODE}_models_pipeline/skipped_targets_stage2.csv"

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"
BUDGET_PREFIX = "budget_"

MAX_N_SPLITS = 5

FEATURE_REDUCTION_TOP_N_FEATURES = 50  # TODO Only applied to regression currently

RANDOM_STATE = 42

MIN_SAMPLES_PER_TARGET = 200
MIN_CLASS_COUNT_FOR_TRAINING = 2  # NOTE Jansen removed, see other comments
MAX_N_SPLITS_CLASSIFICATION = 5

CPU_COUNT = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

# ✔ Select which models to run for classification
# Any subset of {"RF", "LGBM", "XGB", "HGB", "CB"}.
if TASK_MODE == "regression":
    ENABLED_MODELS: List[str] = [  # noqa: UP006
        "RF_REG",
        # "RF_QREG" # predictbudget only -- may not use
    ]
else:
    # ENABLED_MODELS: List[str] = ["RF", "LGBM", "XGB", "HGB", "CB"]  # noqa: UP006
    ENABLED_MODELS: List[str] = ["RF", "LGBM", "XGB", "HGB", "CB"]  # noqa: UP006


# Some models e.g. RandomForestQuantileRegressor do not like numeric NaN
# MODES: None, "median"
if "RF_QREG" in ENABLED_MODELS:
    NUM_IMPUTE = "median"
else:
    NUM_IMPUTE = None

# MLflow (optional)
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "stage2_model_training"

FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SKIPPED_TARGETS_CSV = TRAINED_MODELS_DIR / "skipped_targets_stage2.csv"
CV_RESULTS_CSV = TRAINED_MODELS_DIR / "model_cv_results_parallel.csv"
CV_RESULTS_JSON = TRAINED_MODELS_DIR / "model_cv_results_parallel.json"

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
        isinstance(handler, logging.FileHandler) and getattr(handler, "_stage2_file", False)
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


def detect_columns(df: pd.DataFrame, model_type: str = MODEL_TYPE) -> tuple[list[str], list[str], list[str]]:
    """Detect id_, ft_, y_ columns by prefix."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. Defined types are: {MODEL_TYPES}")
    if model_type == "predictassessment":
        id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        #y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
        y_cols = ['y_csc4a_advccy_freq','y_csc4b_advccy_lvl']
    elif model_type == "predictbudget":
        id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        y_cols = ["budget_total"]
    elif model_type == "assessmentbudget":
        id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
        y_cols = [c for c in df.columns if c.startswith(BUDGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_feature_importances_for_target(
    y_col: str,
    logger: logging.Logger,
) -> Optional[List[str]]:  # noqa: UP006, UP045
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

    # TODO need to discuss how we approach the feature reduction
    selected = df_imp[df_imp["mean_rank"] <= FEATURE_REDUCTION_TOP_N_FEATURES]["feature_name"].astype(str).tolist()

    # if TASK_MODE == "regression":
    #     selected = df_imp[df_imp["mean_rank"] <= FEATURE_REDUCTION_TOP_N_FEATURES]["feature_name"].astype(str).tolist()  # noqa: E501
    # else:
    #     mask = (df_imp[ENABLED_MODELS] > 0).all(axis=1)
    #     selected = df_imp.loc[mask, "feature_name"].astype(str).tolist()

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


def prepare_views_regression(
    features: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: UP006
    """
    Prepare numeric and CatBoost views for regression.

    Only real difference is added imputing for RF Quantile Regressor (which
    we may not end up using)
    """
    numeric = features.copy()
    cat_cols = numeric.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols:
        encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        numeric = encoder.fit_transform(numeric, y)
    elif cat_cols:
        numeric = numeric.drop(columns=cat_cols)

    numeric = numeric.apply(pd.to_numeric, errors="coerce")

    # Impute NaNs if needed:
    if NUM_IMPUTE:
        print(f"Imputing missing values as {NUM_IMPUTE}")
        imputer = SimpleImputer(strategy="median")
        numeric = pd.DataFrame(imputer.fit_transform(numeric), columns=numeric.columns, index=numeric.index)

    cb_view = features.copy()
    cb_cat_cols = cb_view.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        cb_view[col] = cb_view[col].astype("string").fillna(CAT_FILL_VALUE)

    return numeric, cb_view


def prepare_views_classification(
    features: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: UP006
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
) -> Optional[StratifiedKFold]:  # noqa: UP045
    """Choose dynamic StratifiedKFold for classification based on class counts."""
    if TASK_MODE == "regression":
        # Simple
        return KFold(
            n_splits=5,  # Hard-coded
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    else:
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


def build_classification_models(is_binary: bool) -> Dict[str, object]:  # noqa: UP006
    """Build enabled classification models."""
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    models: Dict[str, object] = {}  # noqa: UP006

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


def build_regression_models() -> Dict[str, object]:  # noqa: UP006
    """Build enabled regression models."""
    models: Dict[str, object] = {}  # noqa: UP006

    if "RF_REG" in ENABLED_MODELS:
        models["RF_REG"] = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if "RF_QREG" in ENABLED_MODELS:
        models["RF_QREG"] = RandomForestQuantileRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


# ---------------------------------------------------------------------------
# CV EVALUATION
# ---------------------------------------------------------------------------


def eval_classification_model_cv(
    name: str,
    model_proto: object,
    X_num: pd.DataFrame,  # noqa: N803
    X_cb: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    cv: StratifiedKFold,
    logger: logging.Logger,
) -> Dict[str, float]:  # noqa: UP006
    """Cross-validate one classification model."""
    f1_scores: List[float] = []  # noqa: UP006
    precision_scores: List[float] = []  # noqa: UP006
    recall_scores: List[float] = []  # noqa: UP006
    accuracy_scores: List[float] = []  # noqa: UP006
    auc_scores: List[float] = []  # noqa: UP006
    balanced_accuracy_scores: List[float] = []  # noqa: UP006

    is_binary = y.nunique() == 2

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_num, y), start=1):
        X_train_num = X_num.iloc[train_idx]  # noqa: N806
        X_val_num = X_num.iloc[val_idx]  # noqa: N806
        X_train_cb = X_cb.iloc[train_idx]  # noqa: N806
        X_val_cb = X_cb.iloc[val_idx]  # noqa: N806
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

        balanced_accuracy_scores.append(balanced_accuracy_score(y_val, y_pred))

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
        # Supplementary information on y
        "nbr_classes": len(y.unique()),
        "majority_class_pct": y.value_counts(normalize=True).max(),
        # CV metrics
        "f1_macro": float(np.mean(f1_scores)),
        "precision_macro": float(np.mean(precision_scores)),
        "recall_macro": float(np.mean(recall_scores)),
        "accuracy": float(np.mean(accuracy_scores)),
        "auc": float(np.mean(auc_scores)) if auc_scores else float("nan"),
        "balanced_accuracy": float(np.mean(balanced_accuracy_scores)),
    }


def eval_regression_model_cv(
    name: str,
    model_proto: object,
    X_num: pd.DataFrame,  # noqa: N803
    X_cb: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    cv: KFold,
    logger: logging.Logger,
) -> Dict[str, float]:  # noqa: UP006
    """Cross-validate one regression model."""
    mae_scores: List[float] = []  # noqa: UP006
    mse_scores: List[float] = []  # noqa: UP006
    rmse_scores: List[float] = []  # noqa: UP006
    r2_scores: List[float] = []  # noqa: UP006
    mape_scores: List[float] = []  # noqa: UP006

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_num, y), start=1):  # noqa: B007
        X_train_num = X_num.iloc[train_idx]  # noqa: N806
        X_val_num = X_num.iloc[val_idx]  # noqa: N806

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = model_proto.__class__(**model_proto.get_params())

        model.fit(X_train_num, y_train)
        y_pred = model.predict(X_val_num)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)

        eps = np.finfo(float).eps
        denom = np.maximum(eps, np.abs(y_val))
        mape = float(np.mean(np.abs((y_val - y_pred) / denom)))

        mae_scores.append(mae)
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mape_scores.append(mape)

    return {
        "mae": float(np.mean(mae_scores)),
        "mse": float(np.mean(mse_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "r2": float(np.mean(r2_scores)),
        "mape": float(np.mean(mape_scores)),
    }


# ---------------------------------------------------------------------------
# PER-TARGET PROCESSING
# ---------------------------------------------------------------------------


def process_target(
    y_col: str,
    df: pd.DataFrame,
    ft_cols: List[str],  # noqa: UP006
) -> Dict[str, object]:  # noqa: UP006
    """Process one target for Stage-2 training."""
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
        counts.to_dict() if TASK_MODE == "classification" else None,
    )

    if n_classes < 2:
        logger.warning("Skipping %s: only one class.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "single_class",
            "n_rows": int(n_rows),
        }

    if min_class < MIN_CLASS_COUNT_FOR_TRAINING and TASK_MODE == "classification":
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

    if TASK_MODE == "regression":
        y = y_raw.apply(pd.to_numeric, errors="coerce")
    else:
        # TODO: Don't follow why we do this, so have commented out
        # y = pd.Series(
        #     pd.factorize(y_raw, sort=True)[0],
        #     index=y_raw.index,
        #     dtype="int64",
        # )
        y = y_raw

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

    X = df_target[all_features].copy()  # noqa: N806

    run = None
    if USE_MLFLOW and mlflow_available and mlflow is not None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=f"stage2_{y_col}")
        mlflow.log_param("target", y_col)
        mlflow.log_param("n_rows", int(n_rows))
        mlflow.log_param("n_features", len(all_features))
        mlflow.log_param("enabled_models", ",".join(ENABLED_MODELS))

    if TASK_MODE == "regression":
        X_num, X_cb = prepare_views_regression(X, y)  # noqa: N806
        models = build_regression_models()
    else:
        X_num, X_cb = prepare_views_classification(X, y)  # noqa: N806
        models = build_classification_models(is_binary=y.nunique() == 2)

    if not models:
        logger.warning("No ENABLED_MODELS for %s; skipping.", y_col)
        return {
            "target": y_col,
            "skipped": True,
            "reason": "no_enabled_models",
            "n_rows": int(n_rows),
        }

    model_metrics: Dict[str, Dict[str, float]] = {}  # noqa: UP006
    for name, proto in models.items():
        logger.info("CV for model %s on %s", name, y_col)
        if TASK_MODE == "regression":
            metrics = eval_regression_model_cv(
                name=name,
                model_proto=proto,
                X_num=X_num,
                X_cb=X_cb,
                y=y,
                cv=cv,
                logger=logger,
            )
        else:
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

    metric_best = "f1_macro" if TASK_MODE == "classification" else "rmse"
    best_name = max(
        model_metrics.items(),
        key=lambda kv: kv[1][metric_best],
    )[0]
    logger.info("Best model for %s is %s", y_col, best_name)

    fitted_paths: Dict[str, str] = {}  # noqa: UP006

    # Refit all models on full data.
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
        else:
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)

        out_path = TRAINED_MODELS_DIR / f"{y_col}_{name}.joblib"
        dump(model, out_path)
        fitted_paths[name] = str(out_path)
        logger.info("Saved model %s for %s to %s", name, y_col, out_path)

    best_src_path = fitted_paths[best_name]
    best_model = load(best_src_path)
    best_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    dump(best_model, best_path)
    logger.info("Saved best model alias for %s to %s", y_col, best_path)

    if USE_MLFLOW and mlflow_available and run is not None and mlflow is not None:
        best_metrics = model_metrics[best_name]
        mlflow.log_metrics(best_metrics)
        mlflow.log_param("best_model", best_name)
        mlflow.end_run()

    flat_metrics: Dict[str, float] = {}  # noqa: UP006
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
    """Run Stage-2 model training over all y_* targets."""
    LOG.info("Loading data from %s with row subsetting = %s", fetch_master_data_fn(), DEV_ROW_SUBSET)
    df = return_master_data(add_budget=True, model_type=MODEL_TYPE)
    if "assessment" in MODEL_TYPE or "budget" in MODEL_TYPE:
        LOG.info("Master data filtered to rows where 'id_review_id' not null (i.e the SNAAP sample)")
    LOG.info("Detected %s rows in master data", len(df))

    id_cols, ft_cols, y_cols = detect_columns(df)
    LOG.info(
        "Detected %d id_, %d ft_, %d y_ columns for %s model.", len(id_cols), len(ft_cols), len(y_cols), MODEL_TYPE
    )

    if not ft_cols or not y_cols:
        LOG.error("No ft_ or y_ columns detected; aborting.")
        return

    LOG.info(
        "Starting Stage-2 over %d targets with n_jobs_targets=%d.",
        len(y_cols),
        N_JOBS_TARGETS,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(delayed(process_target)(y_col, df, ft_cols) for y_col in y_cols)

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
