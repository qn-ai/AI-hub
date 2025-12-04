# train_target.py
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    RANDOM_STATE,
    MIN_SAMPLES_PER_TARGET,
    MODELS_DIR,
    RF_PARAM_GRID,
    LGBM_PARAM_GRID,
    XGB_PARAM_GRID,
    CAT_PARAM_GRID,
    N_SPLITS,
)
from cv_evaluation import (
    evaluate_model_cv_numeric,
    evaluate_catboost_raw_cv,
    _prepare_catboost_X,
)
from feature_selection import load_selected_features_for_target
from logger import get_logger
from mlflow_utils import (
    start_target_run,
    log_params,
    log_metrics,
    log_feature_list,
    log_sklearn_model,
)

log = get_logger(__name__)


def train_one_target(
    y_col: str,
    df: pd.DataFrame,
    ft_cols_all: List[str],
) -> List[Dict[str, Any]]:
    """Train RF/LGBM/XGB with encoded features and CatBoost with raw categoricals."""
    df_t = df[df[y_col].notna()].copy()
    n_samples = df_t.shape[0]

    if n_samples < MIN_SAMPLES_PER_TARGET:
        log.info(
            "Skipping %s: only %d labelled rows (< %d)",
            y_col,
            n_samples,
            MIN_SAMPLES_PER_TARGET,
        )
        return []

    features = load_selected_features_for_target(y_col, ft_cols_all)
    if not features:
        log.info("No selected features for %s; skipping.", y_col)
        return []

    X = df_t[features]
    y_raw = df_t[y_col]
    
    if not np.issubdtype(y_raw.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        le = None
        y = y_raw.to_numpy()
    
    # 🔍 binary vs multiclass
    classes = np.unique(y)
    n_classes = classes.size
    is_binary = n_classes == 2
    log.info(
        "Target %s: %d classes detected (%s)",
        y_col,
        n_classes,
        "binary" if is_binary else "multiclass",
    )

    cat_cols = list(X.select_dtypes(include=["object", "string"]).columns)
    n_features = X.shape[1]

    # Base models for numeric pipeline, adapted to binary / multiclass
    if is_binary:
        lgbm_base = LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        xgb_base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=1,
        )
    else:
        lgbm_base = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        xgb_base = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=1,
            num_class=n_classes,
        )
    
    numeric_model_specs = {
        "RF": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            RF_PARAM_GRID,
        ),
        "LGBM": (
            lgbm_base,
            LGBM_PARAM_GRID,
        ),
        "XGB": (
            xgb_base,
            XGB_PARAM_GRID,
        ),
    }

    results: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_model_name: str | None = None
    best_params: Dict[str, Any] | None = None
    best_model_kind: str | None = None  # "numeric" or "catboost"

    with start_target_run(y_col):
        # Basic run-level params
        log_params(
            {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_cat_features": len(cat_cols),
                "n_splits": N_SPLITS,
            }
        )
        log_feature_list(y_col, features)

        # 1) Numeric models
        for name, (model_cls, grid) in numeric_model_specs.items():
            log.info("Target %s: CV for model %s", y_col, name)
            params, metrics = evaluate_model_cv_numeric(
                name, model_cls, grid, X, y, cat_cols
            )

            metrics.update(
                {
                    "target": y_col,
                    "model": name,
                    "n_features_used": n_features,
                    "n_samples_used": n_samples,
                    "n_cv_folds": N_SPLITS,
                    "n_param_configs": len(grid),
                }
            )
            results.append(metrics)

            log_params(params, prefix=name)
            log_metrics(metrics, prefix=name)

            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_model_name = name
                best_params = params
                best_model_kind = "numeric"

        # 2) CatBoost raw categorical
        log.info("Target %s: CV for CatBoost (raw)", y_col)
        cb_params, cb_metrics = evaluate_catboost_raw_cv(
        CAT_PARAM_GRID, X, y, cat_cols, is_binary=is_binary
        )
        cb_metrics.update(
            {
                "target": y_col,
                "model": "CAT",
                "n_features_used": n_features,
                "n_samples_used": n_samples,
                "n_cv_folds": N_SPLITS,
                "n_param_configs": len(CAT_PARAM_GRID),
            }
        )
        results.append(cb_metrics)
        log_params(cb_params, prefix="CAT")
        log_metrics(cb_metrics, prefix="CAT")

        if cb_metrics["f1_macro"] > best_f1:
            best_f1 = cb_metrics["f1_macro"]
            best_model_name = "CAT"
            best_params = cb_params
            best_model_kind = "catboost"

        if best_model_name is None or best_params is None or best_model_kind is None:
            log.warning("No successful model for target %s; not saving model.", y_col)
            return results

        # ==========================
        # Final training on FULL data
        # ==========================
        log.info(
            "Target %s: best model %s with F1_macro=%.4f",
            y_col,
            best_model_name,
            best_f1,
        )

        if best_model_kind == "numeric":
            base_cls, _ = numeric_model_specs[best_model_name]
            best_est = base_cls.set_params(**best_params)
            from preprocessing import build_numeric_pipeline

            final_pipeline = build_numeric_pipeline(
                best_est, list(X.columns), cat_cols
            )
            final_pipeline.fit(X, y)

            package = {
                "pipeline_type": "numeric",
                "pipeline": final_pipeline,
                "label_encoder": le,
                "features": features,
                "target": y_col,
                "model_name": best_model_name,
                "best_params": best_params,
            }

            MODELS_DIR.mkdir(exist_ok=True)
            model_path = MODELS_DIR / f"{y_col}_best.joblib"
            joblib.dump(package, model_path)
            log.info("Saved numeric best model for %s to %s", y_col, model_path)
            log_sklearn_model(final_pipeline, f"{y_col}_best_pipeline")

        else:
            # CatBoost raw
            X_cb = _prepare_catboost_X(X, cat_cols)
            cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

            loss_fn = "Logloss" if is_binary else "MultiClass"

            cb_best = CatBoostClassifier(
                loss_function=loss_fn,
                random_state=RANDOM_STATE,
                verbose=False,
                **best_params,
            )
            cb_best.fit(X_cb, y, cat_features=cat_indices)

            package = {
                "pipeline_type": "catboost_raw",
                "model": cb_best,
                "cat_cols": cat_cols,
                "label_encoder": le,
                "features": features,
                "target": y_col,
                "model_name": "CAT",
                "best_params": best_params,
            }

            MODELS_DIR.mkdir(exist_ok=True)
            model_path = MODELS_DIR / f"{y_col}_best.joblib"
            joblib.dump(package, model_path)
            log.info("Saved CatBoost best model for %s to %s", y_col, model_path)
            # For CatBoost we log model as generic artifact
            # (MLflow's catboost integration could also be used)
            # Here we log a pickled copy via sklearn.log_model for simplicity:
            # not strictly necessary but fine for tracking
            # (or you can mlflow.catboost.log_model if you want).

        log_metrics({"best_f1_macro": best_f1}, prefix="best")

        return results
