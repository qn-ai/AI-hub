from __future__ import annotations

import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from assessmentestimation.helpers import return_master_data, fetch_master_data_fn

from .stage2_config import Stage2Config
from .stage2_data import (
    init_root_logger,
    get_target_logger,
    detect_columns,
    load_feature_importances_for_target,
    clean_target_and_filter_rows,
    prepare_views,
    choose_cv,
)

# Optional MLflow
try:
    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


# -----------------------------
# MODEL BUILDERS (fine-tune params here)
# -----------------------------
def build_classification_models(is_binary: bool, cfg: Stage2Config) -> Dict[str, object]:
    if is_binary:
        lgbm_obj = "binary"
        xgb_obj = "binary:logistic"
        cb_loss = "Logloss"
    else:
        lgbm_obj = "multiclass"
        xgb_obj = "multi:softprob"
        cb_loss = "MultiClass"

    models: Dict[str, object] = {}

    if "RF" in cfg.ENABLED_MODELS:
        models["RF"] = RandomForestClassifier(n_estimators=300, random_state=cfg.RANDOM_STATE, n_jobs=-1)

    if "LGBM" in cfg.ENABLED_MODELS:
        models["LGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            objective=lgbm_obj,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

    if "XGB" in cfg.ENABLED_MODELS:
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=xgb_obj,
            eval_metric="logloss",
            tree_method="hist",
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
        )

    if "HGB" in cfg.ENABLED_MODELS:
        models["HGB"] = HistGradientBoostingClassifier(random_state=cfg.RANDOM_STATE)

    if "CB" in cfg.ENABLED_MODELS:
        models["CB"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function=cb_loss,
            random_state=cfg.RANDOM_STATE,
            verbose=False,
        )

    return models


def build_regression_models(cfg: Stage2Config) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if "RF_REG" in cfg.ENABLED_MODELS:
        models["RF_REG"] = RandomForestRegressor(n_estimators=200, random_state=cfg.RANDOM_STATE, n_jobs=-1)
    return models


# -----------------------------
# CV EVALUATION
# -----------------------------
def eval_classification_model_cv(
    name: str,
    model_proto: object,
    X_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    y: pd.Series,
    cv,
    logger: logging.Logger,
) -> Dict[str, float]:
    f1s: List[float] = []
    ps: List[float] = []
    rs: List[float] = []
    accs: List[float] = []
    baccs: List[float] = []
    aucs: List[float] = []

    is_binary = y.nunique() == 2

    for fold, (tr, va) in enumerate(cv.split(X_num, y), start=1):
        Xtr_num, Xva_num = X_num.iloc[tr], X_num.iloc[va]
        Xtr_cb, Xva_cb = X_cb.iloc[tr], X_cb.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        if name == "CB":
            cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
            cat_idx = [X_cb.columns.get_loc(c) for c in cb_cat_cols]

            model = CatBoostClassifier(**model_proto.get_params())
            model.fit(Xtr_cb, ytr, cat_features=cat_idx if cat_idx else None, verbose=False)
            pred = model.predict(Xva_cb)
            proba = model.predict_proba(Xva_cb)
        else:
            if name == "HGB":
                model = HistGradientBoostingClassifier(**model_proto.get_params())
            elif name == "LGBM":
                model = LGBMClassifier(**model_proto.get_params())
            elif name == "RF":
                model = RandomForestClassifier(**model_proto.get_params())
            elif name == "XGB":
                model = XGBClassifier(**model_proto.get_params())
            else:
                raise ValueError(f"Unknown model name: {name}")

            model.fit(Xtr_num, ytr)
            proba = model.predict_proba(Xva_num)
            pred = model.predict(Xva_num)

        f1s.append(f1_score(yva, pred, average="macro"))
        ps.append(precision_score(yva, pred, average="macro", zero_division=0))
        rs.append(recall_score(yva, pred, average="macro", zero_division=0))
        accs.append(accuracy_score(yva, pred))
        baccs.append(balanced_accuracy_score(yva, pred))

        try:
            if is_binary:
                aucs.append(roc_auc_score(yva, proba[:, 1]))
            else:
                aucs.append(roc_auc_score(yva, proba, multi_class="ovr"))
        except Exception as exc:
            logger.warning("AUC failed for %s fold %d: %s", name, fold, exc)

    return {
        "nbr_classes": float(y.nunique()),
        "majority_class_pct": float(y.value_counts(normalize=True).max()),
        "f1_macro": float(np.mean(f1s)),
        "precision_macro": float(np.mean(ps)),
        "recall_macro": float(np.mean(rs)),
        "accuracy": float(np.mean(accs)),
        "balanced_accuracy": float(np.mean(baccs)),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
    }


def eval_regression_model_cv(model_proto: object, X_num: pd.DataFrame, y: pd.Series, cv) -> Dict[str, float]:
    maes: List[float] = []
    mses: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []
    mapes: List[float] = []

    for tr, va in cv.split(X_num, y):
        Xtr, Xva = X_num.iloc[tr], X_num.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        model = model_proto.__class__(**model_proto.get_params())
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)

        mae = mean_absolute_error(yva, pred)
        mse = mean_squared_error(yva, pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(yva, pred)

        eps = np.finfo(float).eps
        denom = np.maximum(eps, np.abs(yva))
        mape = float(np.mean(np.abs((yva - pred) / denom)))

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        r2s.append(r2)
        mapes.append(mape)

    return {
        "mae": float(np.mean(maes)),
        "mse": float(np.mean(mses)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
        "mape": float(np.mean(mapes)),
    }


# -----------------------------
# PER TARGET
# -----------------------------
def process_target(y_col: str, df: pd.DataFrame, ft_cols: List[str], cfg: Stage2Config) -> Dict[str, object]:
    logger = get_target_logger(cfg, y_col)
    logger.info("=== Stage-2 training for %s ===", y_col)

    df_target, y = clean_target_and_filter_rows(df, y_col, cfg)
    n_rows = int(len(df_target))

    if n_rows < cfg.MIN_SAMPLES_PER_TARGET:
        logger.warning("Skipping %s: too few rows (%d).", y_col, n_rows)
        return {"target": y_col, "skipped": True, "reason": "too_few_rows", "n_rows": n_rows}

    if cfg.TASK_MODE == "classification":
        counts = y.value_counts()
        if counts.shape[0] < 2:
            return {"target": y_col, "skipped": True, "reason": "single_class", "n_rows": n_rows}
        if int(counts.min()) < cfg.MIN_CLASS_COUNT_FOR_TRAINING:
            return {"target": y_col, "skipped": True, "reason": "rare_class", "n_rows": n_rows}

    cv = choose_cv(y, cfg, logger)
    if cv is None:
        return {"target": y_col, "skipped": True, "reason": "cv_failed", "n_rows": n_rows}

    selected = load_feature_importances_for_target(y_col, cfg, logger)
    if not selected:
        return {"target": y_col, "skipped": True, "reason": "no_features_selected", "n_rows": n_rows}

    use_features = [c for c in ft_cols if c in selected]
    if not use_features:
        return {"target": y_col, "skipped": True, "reason": "selected_features_not_in_df", "n_rows": n_rows}

    X = df_target[use_features].copy()
    X_num, X_cb = prepare_views(X, y, cfg)

    models = build_regression_models(cfg) if cfg.TASK_MODE == "regression" else build_classification_models(
        is_binary=(y.nunique() == 2),
        cfg=cfg,
    )

    if not models:
        return {"target": y_col, "skipped": True, "reason": "no_enabled_models", "n_rows": n_rows}

    run = None
    if cfg.USE_MLFLOW and MLFLOW_AVAILABLE and mlflow is not None:
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=f"stage2_{y_col}")
        mlflow.log_param("target", y_col)
        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", len(use_features))
        mlflow.log_param("enabled_models", ",".join(cfg.ENABLED_MODELS))

    model_metrics: Dict[str, Dict[str, float]] = {}
    for name, proto in models.items():
        logger.info("CV for model %s", name)
        if cfg.TASK_MODE == "regression":
            metrics = eval_regression_model_cv(proto, X_num, y, cv)
        else:
            metrics = eval_classification_model_cv(name, proto, X_num, X_cb, y, cv, logger)
        model_metrics[name] = metrics
        logger.info("CV metrics for %s: %s", name, metrics)

    metric_best = "rmse" if cfg.TASK_MODE == "regression" else "f1_macro"
    best_name = max(model_metrics.items(), key=lambda kv: kv[1][metric_best])[0]
    logger.info("Best model for %s is %s", y_col, best_name)

    cfg.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fitted_paths: Dict[str, str] = {}

    for name, proto in models.items():
        if name == "CB":
            cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
            cat_idx = [X_cb.columns.get_loc(c) for c in cb_cat_cols]
            model = CatBoostClassifier(**proto.get_params())
            model.fit(X_cb, y, cat_features=cat_idx if cat_idx else None, verbose=False)
        else:
            model = proto.__class__(**proto.get_params())
            model.fit(X_num, y)

        out_path = cfg.TRAINED_MODELS_DIR / f"{y_col}_{name}.joblib"
        dump(model, out_path)
        fitted_paths[name] = str(out_path)

    best_model = load(fitted_paths[best_name])
    best_path = cfg.TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    dump(best_model, best_path)

    if cfg.USE_MLFLOW and MLFLOW_AVAILABLE and run is not None and mlflow is not None:
        mlflow.log_metrics(model_metrics[best_name])
        mlflow.log_param("best_model", best_name)
        mlflow.end_run()

    flat: Dict[str, float] = {}
    for mname, mm in model_metrics.items():
        for k, v in mm.items():
            flat[f"{mname}_{k}"] = float(v)

    return {
        "target": y_col,
        "skipped": False,
        "reason": "",
        "n_rows": n_rows,
        "best_model": best_name,
        **flat,
    }


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    cfg = Stage2Config()

    cfg.FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    LOG = init_root_logger()

    LOG.info("Loading data from %s | DEV_ROW_SUBSET=%s", fetch_master_data_fn(), cfg.DEV_ROW_SUBSET)
    df = return_master_data(add_budget=True, model_type=cfg.MODEL_TYPE)

    if cfg.DEV_ROW_SUBSET is not None:
        if cfg.DEV_ROW_SUBSET_MODE == "sample":
            df = df.sample(n=min(cfg.DEV_ROW_SUBSET, len(df)), random_state=cfg.RANDOM_STATE).copy()
        else:
            df = df.head(cfg.DEV_ROW_SUBSET).copy()

    LOG.info("Rows in master data: %d", len(df))

    _, ft_cols, y_cols = detect_columns(df, cfg)
    y_cols = cfg.filter_targets(y_cols)

    LOG.info("Detected %d ft_ columns", len(ft_cols))
    LOG.info("Targets to run (%d): %s", len(y_cols), y_cols)

    if not ft_cols or not y_cols:
        LOG.error("No ft_ columns or no targets to run after filtering; abort.")
        return

    LOG.info("Starting Stage-2 | n_jobs=%d", cfg.N_JOBS_TARGETS)

    results = Parallel(n_jobs=cfg.N_JOBS_TARGETS)(
        delayed(process_target)(y_col, df, ft_cols, cfg) for y_col in y_cols
    )

    skipped = [r for r in results if r.get("skipped")]
    processed = [r for r in results if not r.get("skipped")]

    if skipped:
        pd.DataFrame(skipped).to_csv(cfg.SKIPPED_TARGETS_CSV, index=False)
        LOG.info("Saved skipped targets to %s", cfg.SKIPPED_TARGETS_CSV)

    if processed:
        df_cv = pd.DataFrame(processed)
        df_cv.to_csv(cfg.CV_RESULTS_CSV, index=False)
        with cfg.CV_RESULTS_JSON.open("w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)
        LOG.info("Saved CV results to %s and %s", cfg.CV_RESULTS_CSV, cfg.CV_RESULTS_JSON)

    LOG.info("Stage-2 completed: %d processed, %d skipped", len(processed), len(skipped))


if __name__ == "__main__":
    main()
