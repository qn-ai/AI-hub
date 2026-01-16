"""
Stage-3: Parallel, chunked prediction on a new dataset (classification only).

Modes
-----

PREDICTION_MODE = "best"
    - For each target y_<name>, loads y_<name>_best.joblib.
    - Outputs:
        y_<name>                              (actual, if present)
        y_<name>_interpolated_model4          (predicted label)
        y_<name>_interpolated_model4_metric1  (row-level probability)
        y_<name>_interpolated_model4_metric2_f1
        y_<name>_interpolated_model4_metric3_recall
        y_<name>_interpolated_model4_metric4_precision
        y_<name>_interpolated_model4_metric5_auc

PREDICTION_MODE = "all_models"
    - For each target y_<name>, attempts to load, for each enabled model M:
        y_<name>_M.joblib
      and outputs:
        y_<name>                              (actual, once)
        y_<name>_M_interpolated_model4
        y_<name>_M_interpolated_model4_metric1
        y_<name>_M_interpolated_model4_metric2_f1
        y_<name>_M_interpolated_model4_metric3_recall
        y_<name>_M_interpolated_model4_metric4_precision
        y_<name>_M_interpolated_model4_metric5_auc

Global metrics are pulled from Stage-2 CSV (model_cv_results_parallel.csv):

    - For the best model: F1/Recall/Precision/AUC populated.
    - For other models: metrics populated if present in CSV,
      otherwise NaN with a warning.

Inputs
------
- new_data.csv
- feature_importances/feature_importances_<y>.csv
- trained_models/y_<target>_<MODEL>.joblib
- trained_models/y_<target>_best.joblib
- trained_models/model_cv_results_parallel.csv

Output
------
- stage3_predictions.csv with id_pwd_id and all prediction columns.

Logs
----
- logs/y_<target>_stage3.log per target.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

from joblib import Parallel, delayed, load
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import XGBClassifier
from quantile_forest import RandomForestQuantileRegressor


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier

import pytz
import datetime

from src.assessmentestimation.helpers import return_master_data, fetch_master_data_fn
from src.assessmentestimation.scoring_output_helpers import validate_df, recombine_multi_select

# Optional MLflow (often off for Stage-3)
try:
    import mlflow

    mlflow_available = True
except Exception:  # pragma: no cover - optional
    mlflow = None  # type: ignore[assignment]
    mlflow_available = False


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Global task mode:
#   "classification" → all targets treated as classification
#   "regression"     → all targets treated as regression (numeric only)
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

NEW_DATA_PATH = "new_data.csv"

FEATURE_IMPORTANCE_DIR = Path(f"ae_models_pipeline/{MODEL_TYPE}/feature_importances")
TRAINED_MODELS_DIR = Path(f"ae_models_pipeline/{MODEL_TYPE}/trained_models")
STAGE2_METRICS_PATH = f"ae_models_pipeline/{MODEL_TYPE}/model_cv_results_parallel.csv"
LOG_DIR = Path(f"ae_models_pipeline/{MODEL_TYPE}/logs")

CHECKS_OUTPUT_JSON = Path(f"ae_models_pipeline/{MODEL_TYPE}/stage_3_checks.json")

ID_COL = "id_pwd_id"

ID_PREFIX = "id_"
FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"
BUDGET_PREFIX = "budget_"

RANDOM_STATE = 42

CHUNK_SIZE = 50_000
CPU_COUNT = os.cpu_count() or 4
N_JOBS_TARGETS = max(min(CPU_COUNT - 1, 16), 2)

USE_CATBOOST_ENCODER = True
CAT_FILL_VALUE = "NA_CAT"

FEATURE_REDUCTION_TOP_N_FEATURES = 50

# ✔ Models we consider for prediction when PREDICTION_MODE="all_models"
if TASK_MODE == "regression":
    ENABLED_MODELS: List[str] = [  # noqa: UP006
        "RF_REG",
        # "RF_QREG" # predictbudget only -- may not use
    ]
else:
    ENABLED_MODELS: List[str] = ["RF"]  # noqa: UP006
    # ENABLED_MODELS: List[str] = ["RF", "LGBM", "XGB", "HGB", "CB"]  # noqa: UP006

# Prediction mode: "best" or "all_models"
PREDICTION_MODE = "best"

MODEL_SUFFIX = "modelBaselineRF"

# Some models e.g. RandomForestQuantileRegressor do not like numeric NaN
# MODES: None, "median"
if "RF_QREG" in [ENABLED_MODELS]:
    NUM_IMPUTE = "median"
else:
    NUM_IMPUTE = None

# MLflow (optional)
USE_MLFLOW = False
MLFLOW_EXPERIMENT_NAME = "stage3_scoring"

PREDICTIONS_OUTPUT_NAME = f"stage3_predictions_{MODEL_TYPE}_{MODEL_SUFFIX}"
CV_RESULTS_CSV = TRAINED_MODELS_DIR / "model_cv_results_parallel.csv"

FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Modes: None, [0.25, 0.75], [0.05, 0.95]
# (in development, also won't work for all model types)
if "RF_QREG" in ENABLED_MODELS:
    RETURN_PREDICTION_INTERVAL = [0.25, 0.75]
else:
    RETURN_PREDICTION_INTERVAL = None

# For simulating from class probabilities
RETURN_CLASS_PROBABILITY_DICT = True

RETURN_CONFIDENCE_METRICS = False

WRITE_TO_S3 = False
S3_DIRECTORY = "session-data/ada-bm-assessment-estimation/data/predictions"

# Apparently BMcal still needs the multi select cols to be recombined.....
RECOMBINE_MULTI_SELECT = True

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("stage3")


def get_target_logger(y_col: str) -> logging.Logger:
    """Create a per-target logger writing to logs/y_<target>_stage3.log."""
    logger = logging.getLogger(f"stage3.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "_stage3_file", False)
        for handler in logger.handlers
    )
    if not exists:
        file_handler = logging.FileHandler(
            LOG_DIR / f"{y_col}_stage3.log",
            mode="w",
            encoding="utf-8",
        )
        file_handler._stage3_file = True  # type: ignore[attr-defined]
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
        y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    elif model_type == "predictbudget":
        id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
        y_cols = ["budget_total"]
    elif model_type == "assessmentbudget":
        id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
        y_cols = [c for c in df.columns if c.startswith(BUDGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_cv_results() -> Optional[pd.DataFrame]:  # noqa: UP045
    """Load Stage-2 CV results if available."""
    if not CV_RESULTS_CSV.exists():
        LOG.warning("Stage-2 CV results not found at %s", CV_RESULTS_CSV)
        return None
    return pd.read_csv(CV_RESULTS_CSV)


def get_cv_row_for_target(
    df_cv: Optional[pd.DataFrame],  # noqa: UP045
    y_col: str,
) -> Optional[pd.Series]:  # noqa: UP045
    """Return the CV results row for a given target."""
    if df_cv is None:
        return None
    subset = df_cv[df_cv["target"] == y_col]
    if subset.empty:
        return None
    return subset.iloc[0]


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


def prepare_views_classification_for_prediction(
    X: pd.DataFrame,  # noqa: N803
    y_optional: Optional[pd.Series],  # noqa: UP045
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: UP006
    """Prepare numeric + CatBoost views for classification predictions."""
    X_num = X.copy()  # noqa: N806
    cat_cols = X_num.select_dtypes(include=["object"]).columns.tolist()

    if USE_CATBOOST_ENCODER and cat_cols and y_optional is not None:
        mask = y_optional.notna()
        if mask.any():
            logger.info(
                "Fitting CatBoostEncoder on %d labeled rows for prediction.",
                int(mask.sum()),
            )
            encoder = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
            encoder = encoder.fit(X_num.loc[mask, :], y_optional.loc[mask])
            X_num = encoder.transform(X_num)  # noqa: N806
        else:
            logger.info(
                "No labeled rows for this target in Stage-3; dropping object columns for numeric view.",
            )
            X_num = X_num.drop(columns=cat_cols)  # noqa: N806
    elif cat_cols:
        logger.info(
            "No labels or CatBoostEncoder disabled; dropping object columns.",
        )
        X_num = X_num.drop(columns=cat_cols)  # noqa: N806

    X_num = X_num.apply(pd.to_numeric, errors="coerce")  # noqa: N806

    X_cb = X.copy()  # noqa: N806
    cb_cat_cols = X_cb.select_dtypes(include=["object"]).columns.tolist()
    for col in cb_cat_cols:
        X_cb[col] = X_cb[col].astype("string").fillna(CAT_FILL_VALUE)

    return X_num, X_cb


def prepare_views_regression_for_prediction(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:  # noqa: N803
    """
    Encoding.

    CatBoost encoder for categoric cols, transforms rows with y unknown as well
    Numeric untouched, impute NA if necessary
    """
    X2 = X.copy()  # noqa: N806
    cat_cols = X2.select_dtypes(include=["object"]).columns.tolist()

    mask_known = y.notna()
    X_known = X.loc[mask_known].copy()  # noqa: N806
    y_known = y.loc[mask_known]
    X_unknown = X.loc[~mask_known].copy()  # noqa: N806

    if USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=RANDOM_STATE)
        X_known_enc = enc.fit_transform(X_known, y_known)  # noqa: N806

        if not X_unknown.empty:
            X_unknown_enc = enc.transform(X_unknown)  # noqa: N806
            X_encoded = pd.concat([X_known_enc, X_unknown_enc], axis=0).sort_index()  # noqa: N806
        else:
            X_encoded = X_known_enc.sort_index()  # noqa: N806

    elif cat_cols:
        X_encoded = X2.drop(columns=cat_cols)  # noqa: N806

    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")  # noqa: N806

    # Impute NaNs if needed:
    if NUM_IMPUTE:
        print(f"Imputing missing values as {NUM_IMPUTE}")
        imputer = SimpleImputer(strategy="median")
        X_encoded = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns, index=X_encoded.index)  # noqa: N806

    return X_encoded


def available_models(y_col: str) -> Dict[str, Path]:  # noqa: UP006
    """Return available model paths for a target."""
    paths: Dict[str, Path] = {}  # noqa: UP006
    for name in ENABLED_MODELS:
        path = TRAINED_MODELS_DIR / f"{y_col}_{name}.joblib"
        if path.exists():
            paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# PER-TARGET PREDICTION
# ---------------------------------------------------------------------------


def predict_for_target(
    y_col: str,
    df: pd.DataFrame,
    id_cols: List[str],  # noqa: UP006
    base_ft_cols: List[str],  # noqa: UP006
    df_cv: Optional[pd.DataFrame],  # noqa: UP045
) -> Optional[pd.DataFrame]:  # noqa: UP045
    """Run Stage-3 scoring for a single target (classification only)."""
    logger = get_target_logger(y_col)
    logger.info(
        "=== Stage-3 scoring for %s (prediction_mode=%s) ===",
        y_col,
        PREDICTION_MODE,
    )

    selected = load_feature_importances_for_target(y_col=y_col, logger=logger)
    if not selected:
        return None

    use_features = [c for c in base_ft_cols if c in selected and c in df.columns]
    if not use_features:
        logger.warning(
            "No usable features for %s in new_data; skipping.",
            y_col,
        )
        return None

    X_full = df[use_features].copy()  # noqa: N806
    y_actual: Optional[pd.Series]  # noqa: UP045
    if y_col in df.columns:
        y_actual = df[y_col]
    else:
        y_actual = None

    cv_row = get_cv_row_for_target(df_cv, y_col)

    best_model_path = TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    model_paths = available_models(y_col)

    if PREDICTION_MODE == "best":
        if not best_model_path.exists():
            logger.warning(
                "Prediction mode 'best' but best model missing for %s; skipping.",
                y_col,
            )
            return None
        model_paths = {}  # we only use best
    else:
        if not model_paths and not best_model_path.exists():
            logger.warning(
                "No individual or best models for %s; skipping.",
                y_col,
            )
            return None

    if TASK_MODE == "regression":
        X_num = prepare_views_regression_for_prediction(  # noqa: N806
            X=X_full,
            y=y_actual,
        )
    else:
        X_num, X_cb = prepare_views_classification_for_prediction(  # noqa: N806
            X_full,
            y_optional=y_actual,
            logger=logger,
        )

    n_rows = df.shape[0]
    chunks = list(range(0, n_rows, CHUNK_SIZE))

    base_out = pd.DataFrame(index=df.index)
    for col in [
        "id_review_id",
        "id_pwd_id",
    ]:
        if col in df.columns:
            base_out[col] = df[col]

    if y_actual is not None:
        base_out[y_col] = y_actual

    # ------------------------------------------------------------------ #
    # Mode: BEST
    # ------------------------------------------------------------------ #
    if PREDICTION_MODE == "best":
        model = load(best_model_path)
        all_preds: List[np.ndarray] = []  # noqa: UP006
        all_probs: List[np.ndarray] = []  # noqa: UP006
        all_preds_lower: List[np.ndarray] = []  # noqa: UP006
        all_preds_upper: List[np.ndarray] = []  # noqa: UP006

        for start in chunks:
            end = min(start + CHUNK_SIZE, n_rows)
            idx = X_num.index[start:end]
            if isinstance(model, CatBoostClassifier):
                preds = model.predict(X_cb.loc[idx, :])
                proba = model.predict_proba(X_cb.loc[idx, :])
            elif isinstance(model, HistGradientBoostingClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = np.argmax(proba, axis=1)
            elif isinstance(model, LGBMClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = np.argmax(proba, axis=1)
            elif isinstance(model, RandomForestClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = model.predict(X_num.loc[idx, :])
            elif isinstance(model, XGBClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = np.argmax(proba, axis=1)
            elif isinstance(model, RandomForestRegressor):
                proba = np.full(len(idx), np.nan)
                preds = model.predict(X_num.loc[idx, :])
            elif isinstance(model, RandomForestQuantileRegressor):
                proba = np.full(len(idx), np.nan)
                preds = model.predict(X_num.loc[idx, :])

                if RETURN_PREDICTION_INTERVAL:
                    pred_lower = model.predict(X_num.loc[idx, :], RETURN_PREDICTION_INTERVAL[0])
                    pred_upper = model.predict(X_num.loc[idx, :], RETURN_PREDICTION_INTERVAL[1])

            else:
                raise ValueError(
                    f"Unsupported best model type for {y_col}: {type(model)}",
                )

            all_preds.append(preds)
            all_probs.append(proba)
            if RETURN_PREDICTION_INTERVAL:
                all_preds_lower.append(pred_lower)
                all_preds_upper.append(pred_upper)

        y_pred = np.concatenate(all_preds)
        proba_all = np.concatenate(all_probs, axis=0)
        max_proba = np.array([i.max() for i in proba_all])

        classes = model.classes_
        proba_dict = [
            {cls: f"{val:.2f}" for cls, val in zip(classes, row)} 
            for row in proba_all
        ]

        if RETURN_PREDICTION_INTERVAL:
            pred_lower = np.concatenate(all_preds_lower)
            pred_upper = np.concatenate(all_preds_upper)

        out = base_out.copy()
        base = f"{y_col}_{MODEL_SUFFIX}"
        out[f"{base}_predicted"] = y_pred

        if RETURN_PREDICTION_INTERVAL:
            out[f"{base}_pred_lower"] = pred_lower
            out[f"{base}_pred_upper"] = pred_upper

        if RETURN_CLASS_PROBABILITY_DICT:
            out[f"{base}_class_probability"] = proba_dict

        if cv_row is not None:
            best_name = cv_row.get("best_model")
            if isinstance(best_name, float) and np.isnan(best_name):
                best_name = None
        else:
            best_name = None

        if RETURN_CONFIDENCE_METRICS:
            if TASK_MODE == "regression":
                prefix = f"{best_name}_"
                mae_val = cv_row.get(f"{prefix}mae", np.nan)
                mse_val = cv_row.get(f"{prefix}mse", np.nan)
                rmse_val = cv_row.get(f"{prefix}rmse", np.nan)

                out[f"{base}_metric2_mae"] = mae_val
                out[f"{base}_metric3_mse"] = mse_val
                out[f"{base}_metric4_rmse"] = rmse_val
            else:
                if cv_row is not None and best_name is not None:
                    prefix = f"{best_name}_"
                    f1_val = cv_row.get(f"{prefix}f1_macro", np.nan)
                    rec_val = cv_row.get(f"{prefix}recall_macro", np.nan)
                    prec_val = cv_row.get(f"{prefix}precision_macro", np.nan)
                    acc_val = cv_row.get(f"{prefix}accuracy", np.nan)
                    auc_val = cv_row.get(f"{prefix}auc", np.nan)
                else:
                    f1_val = rec_val = prec_val = acc_val = auc_val = np.nan

                out[f"{base}_metric1"] = max_proba
                out[f"{base}_metric2_f1"] = f1_val
                out[f"{base}_metric3_recall"] = rec_val
                out[f"{base}_metric4_precision"] = prec_val
                out[f"{base}_metric5_auc"] = auc_val

        logger.info("Finished Stage-3 scoring for %s", y_col)
        return out

    # ------------------------------------------------------------------ #
    # Mode: ALL MODELS
    # # Note: Not updated for regression -- just not needed
    # ------------------------------------------------------------------ #
    models_available = model_paths
    if not models_available and best_model_path.exists():
        models_available = {"BEST": best_model_path}
    if not models_available:
        logger.warning("No models available for %s; skipping.", y_col)
        return None

    out = base_out.copy()

    for name, path in models_available.items():
        logger.info("Scoring %s with model %s", y_col, name)
        model = load(path)
        all_preds: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        for start in chunks:
            end = min(start + CHUNK_SIZE, n_rows)
            idx = X_num.index[start:end]

            if isinstance(model, CatBoostClassifier):
                preds = model.predict(X_cb.loc[idx, :])
                proba = model.predict_proba(X_cb.loc[idx, :])
            elif isinstance(model, HistGradientBoostingClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = np.argmax(proba, axis=1)
            elif isinstance(model, LGBMClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = np.argmax(proba, axis=1)
            elif isinstance(model, RandomForestClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = model.predict(X_num.loc[idx, :])
            elif isinstance(model, XGBClassifier):
                proba = model.predict_proba(X_num.loc[idx, :])
                preds = np.argmax(proba, axis=1)
            else:
                raise ValueError(
                    f"Unsupported classification model type for {y_col}: {type(model)}",
                )

            all_preds.append(preds)
            all_probs.append(proba)

        y_pred = np.concatenate(all_preds)
        proba_all = np.concatenate(all_probs, axis=0)
        max_proba = proba_all.max(axis=1)

        prefix_cols = f"{y_col}_{name}_interpolated_{MODEL_SUFFIX}"
        out[prefix_cols] = y_pred
        out[f"{prefix_cols}_metric1"] = max_proba

        if cv_row is not None and name in ["RF", "LGBM", "XGB", "HGB", "CB"]:
            metric_prefix = f"{name}_"
            f1_val = cv_row.get(f"{metric_prefix}f1_macro", np.nan)
            rec_val = cv_row.get(f"{metric_prefix}recall_macro", np.nan)
            prec_val = cv_row.get(f"{metric_prefix}precision_macro", np.nan)
            acc_val = cv_row.get(f"{metric_prefix}accuracy", np.nan)
            auc_val = cv_row.get(f"{metric_prefix}auc", np.nan)
        else:
            f1_val = rec_val = prec_val = acc_val = np.nan  # noqa: F841

        out[f"{prefix_cols}_metric2_f1"] = f1_val
        out[f"{prefix_cols}_metric3_recall"] = rec_val
        out[f"{prefix_cols}_metric4_precision"] = prec_val
        out[f"{prefix_cols}_metric5_auc"] = auc_val

    logger.info("Finished Stage-3 scoring for %s", y_col)
    return out


def postprocessing(df):  # noqa: ANN001, ANN201
    """Return desired output format."""
    df.insert(0, "snaap_sample", df["id_review_id"].notnull())
    df.drop("id_review_id", inplace=True, axis=1)
    df.columns = df.columns.str.replace(r"^(?:y_|id_)", "", regex=True)

    return df


def write_to_s3(local_path: str, s3_dir: str) -> None:
    """
    Write output to S3.

    TODO: Use the DaSH python code, I don't know where it is
    """
    import rpy2.robjects as ro  # noqa: F401
    from rpy2.robjects import r

    sydney_tz = pytz.timezone("Australia/Sydney")
    datetimestamp = datetime.datetime.now(sydney_tz).strftime("%Y%m%d%H%M")

    s3_path = f"{s3_dir}/AEst_predictions_{MODEL_SUFFIX}_{datetimestamp}.csv"

    r.assign("file_path", local_path)
    r.assign("s3_file_path", s3_path)

    r("actutils::s3_upload(file = file_path, to = s3_file_path )")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Stage-3 predictions over all y_* targets on new_data.csv."""
    LOG.info("Loading data from %s with row subsetting = %s", fetch_master_data_fn(), DEV_ROW_SUBSET)
    df = return_master_data(add_budget=True, model_type=MODEL_TYPE, scoring_or_training="scoring")

    id_cols, ft_cols, _ = detect_columns(df)
    LOG.info(
        "Detected %d id_, %d ft_ columns in new_data.",
        len(id_cols),
        len(ft_cols),
    )

    df_cv = load_cv_results()

    if USE_MLFLOW and mlflow_available and mlflow is not None:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model_best_files = list(TRAINED_MODELS_DIR.glob("*_best.joblib"))
    targets_from_models = sorted(
        {p.name.split("_best.joblib")[0] for p in model_best_files},
    )

    if not targets_from_models:
        LOG.error("No y_*_best.joblib models found in %s; aborting Stage-3.", TRAINED_MODELS_DIR)
        return

    LOG.info(
        "Stage-3 will score %d targets with n_jobs_targets=%d (mode=%s).",
        len(targets_from_models),
        N_JOBS_TARGETS,
        PREDICTION_MODE,
    )

    results = Parallel(n_jobs=N_JOBS_TARGETS)(
        delayed(predict_for_target)(
            y_col,
            df,
            id_cols,
            ft_cols,
            df_cv,
        )
        for y_col in targets_from_models
    )

    frames = [r for r in results if r is not None]
    if not frames:
        LOG.error("No predictions generated in Stage-3.")
        return

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(
            frame.drop(columns=id_cols, errors="ignore"),
            how="outer",
        )

    combined = postprocessing(combined)

    checks_dict = validate_df(output=combined, master_data=df, MODEL_SUFFIX=MODEL_SUFFIX)
    with CHECKS_OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(checks_dict, f, indent=2, default=str)

    write_path = Path(f"{PREDICTIONS_OUTPUT_NAME}.csv")
    combined.to_csv(write_path, index=False)
    LOG.info("Stage-3 predictions saved to %s", write_path)

    if WRITE_TO_S3:
        write_to_s3(local_path=str(write_path), s3_dir=S3_DIRECTORY)

    # 20251223 BMcal currently needs multi-select combined into one col
    if RECOMBINE_MULTI_SELECT:
        combined = recombine_multi_select(combined, MODEL_SUFFIX)

        write_path_combined = Path(f"{PREDICTIONS_OUTPUT_NAME}_comb_multi.csv")
        combined.to_csv(write_path, index=False)

        if WRITE_TO_S3:
            write_to_s3(local_path=str(write_path_combined), s3_dir=S3_DIRECTORY)

if __name__ == "__main__":
    main()
