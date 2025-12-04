# cv_evaluation.py
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import time
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import RANDOM_STATE, N_SPLITS
from preprocessing import build_numeric_pipeline


def evaluate_model_cv_numeric(
    model_name: str,
    base_model,
    param_grid: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """CV for RF/LGBM/XGB using numeric pipeline (CatBoostEncoder + imputer)."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    feature_names = list(X.columns)
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, float] = {"f1_macro": -np.inf}
    best_time = 0.0

    for params in param_grid:
        fold_metrics = {
            "f1_macro": [],
            "precision_macro": [],
            "recall_macro": [],
            "accuracy": [],
            "roc_auc_macro_ovr": [],
            "log_loss": [],
        }
        total_time = 0.0

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = base_model.set_params(**params)
            pipe = build_numeric_pipeline(model, feature_names, cat_cols)

            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            total_time += time.perf_counter() - t0

            y_pred = pipe.predict(X_val)
            try:
                y_proba = pipe.predict_proba(X_val)
            except Exception:
                y_proba = None

            fold_metrics["f1_macro"].append(
                f1_score(y_val, y_pred, average="macro")
            )
            fold_metrics["precision_macro"].append(
                precision_score(y_val, y_pred, average="macro", zero_division=0)
            )
            fold_metrics["recall_macro"].append(
                recall_score(y_val, y_pred, average="macro", zero_division=0)
            )
            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))

            if y_proba is not None:
                try:
                    auc = roc_auc_score(
                        y_val, y_proba, average="macro", multi_class="ovr"
                    )
                except Exception:
                    auc = np.nan
                fold_metrics["roc_auc_macro_ovr"].append(auc)

                try:
                    ll = log_loss(y_val, y_proba)
                except Exception:
                    ll = np.nan
                fold_metrics["log_loss"].append(ll)
            else:
                fold_metrics["roc_auc_macro_ovr"].append(np.nan)
                fold_metrics["log_loss"].append(np.nan)

        mean_metrics = {k: float(np.nanmean(v)) for k, v in fold_metrics.items()}
        mean_metrics["train_time_total_sec"] = float(total_time)

        if mean_metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_metrics = mean_metrics
            best_params = params
            best_time = total_time

    best_metrics["train_time_total_sec"] = float(best_time)
    return best_params, best_metrics


def _prepare_catboost_X(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Prepare X for CatBoost raw categorical training."""
    X_cb = X.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype("string").fillna("NA_CAT")
    return X_cb


def evaluate_catboost_raw_cv(
    param_grid: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """CV for CatBoost using raw categoricals and native NaN handling."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    X_cb = _prepare_catboost_X(X, cat_cols)
    cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, float] = {"f1_macro": -np.inf}
    best_time = 0.0

    for params in param_grid:
        fold_metrics = {
            "f1_macro": [],
            "precision_macro": [],
            "recall_macro": [],
            "accuracy": [],
            "roc_auc_macro_ovr": [],
            "log_loss": [],
        }
        total_time = 0.0

        for train_idx, val_idx in skf.split(X_cb, y):
            X_train, X_val = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = CatBoostClassifier(
                loss_function="MultiClass",
                random_state=RANDOM_STATE,
                verbose=False,
                **params,
            )

            t0 = time.perf_counter()
            model.fit(X_train, y_train, cat_features=cat_indices)
            total_time += time.perf_counter() - t0

            y_pred = model.predict(X_val)
            try:
                y_proba = model.predict_proba(X_val)
            except Exception:
                y_proba = None

            fold_metrics["f1_macro"].append(
                f1_score(y_val, y_pred, average="macro")
            )
            fold_metrics["precision_macro"].append(
                precision_score(y_val, y_pred, average="macro", zero_division=0)
            )
            fold_metrics["recall_macro"].append(
                recall_score(y_val, y_pred, average="macro", zero_division=0)
            )
            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))

            if y_proba is not None:
                try:
                    auc = roc_auc_score(
                        y_val, y_proba, average="macro", multi_class="ovr"
                    )
                except Exception:
                    auc = np.nan
                fold_metrics["roc_auc_macro_ovr"].append(auc)

                try:
                    ll = log_loss(y_val, y_proba)
                except Exception:
                    ll = np.nan
                fold_metrics["log_loss"].append(ll)
            else:
                fold_metrics["roc_auc_macro_ovr"].append(np.nan)
                fold_metrics["log_loss"].append(np.nan)

        mean_metrics = {k: float(np.nanmean(v)) for k, v in fold_metrics.items()}
        mean_metrics["train_time_total_sec"] = float(total_time)

        if mean_metrics["f1_macro"] > best_metrics["f1_macro"]:
            best_metrics = mean_metrics
            best_params = params
            best_time = total_time

    best_metrics["train_time_total_sec"] = float(best_time)
    return best_params, best_metrics
