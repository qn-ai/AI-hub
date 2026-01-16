from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from src.assessmentestimation.helpers import return_master_data, fetch_master_data_fn
from src.assessmentestimation.scoring_output_helpers import validate_df, recombine_multi_select

from .stage3_config import Stage3Config
from .stage3_data import (
    init_root_logger,
    get_target_logger,
    detect_columns,
    load_stage2_cv_results,
    get_cv_row_for_target,
    load_feature_importances_for_target,
    prepare_views_for_prediction,
    load_label_map,
    decode_predictions,
    proba_to_dict_list,
    available_model_paths,
)


def _base_output(df: pd.DataFrame, id_cols: List[str], y_actual: Optional[pd.Series], y_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in id_cols:
        if c in df.columns:
            out[c] = df[c]
    if y_actual is not None:
        out[y_col] = y_actual
    return out


def _chunk_ranges(n: int, chunk: int) -> List[tuple[int, int]]:
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def _predict_chunk(model, X_num_chunk: pd.DataFrame, X_cb_chunk: pd.DataFrame):
    if isinstance(model, CatBoostClassifier):
        pred = model.predict(X_cb_chunk)
        proba = model.predict_proba(X_cb_chunk)
        classes = getattr(model, "classes_", None)
        return pred, proba, classes

    if isinstance(model, HistGradientBoostingClassifier):
        proba = model.predict_proba(X_num_chunk)
        pred = np.argmax(proba, axis=1)
        return pred, proba, None

    if isinstance(model, LGBMClassifier):
        proba = model.predict_proba(X_num_chunk)
        pred = np.argmax(proba, axis=1)
        return pred, proba, None

    if isinstance(model, RandomForestClassifier):
        proba = model.predict_proba(X_num_chunk)
        pred = model.predict(X_num_chunk)
        classes = getattr(model, "classes_", None)
        return pred, proba, classes

    if isinstance(model, XGBClassifier):
        proba = model.predict_proba(X_num_chunk)
        pred = np.argmax(proba, axis=1)
        return pred, proba, None

    if isinstance(model, (RandomForestRegressor,)):
        pred = model.predict(X_num_chunk)
        proba = None
        return pred, proba, None

    raise ValueError(f"Unsupported model type: {type(model)}")


def predict_for_target(
    y_col: str,
    df: pd.DataFrame,
    id_cols: List[str],
    ft_cols: List[str],
    df_cv: Optional[pd.DataFrame],
    cfg: Stage3Config,
) -> Optional[pd.DataFrame]:
    logger = get_target_logger(cfg, y_col)
    logger.info("=== Stage-3 scoring for %s | mode=%s ===", y_col, cfg.PREDICTION_MODE)

    selected = load_feature_importances_for_target(cfg, y_col, logger)
    if not selected:
        return None

    use_features = [c for c in ft_cols if c in selected and c in df.columns]
    if not use_features:
        logger.warning("No usable features for %s in scoring data; skipping.", y_col)
        return None

    X_full = df[use_features].copy()
    y_actual = df[y_col] if y_col in df.columns else None

    cv_row = get_cv_row_for_target(df_cv, y_col)
    best_name = None
    if cv_row is not None:
        bn = cv_row.get("best_model", None)
        if isinstance(bn, str) and bn:
            best_name = bn

    X_num, X_cb = prepare_views_for_prediction(X_full, y_actual, cfg, logger)
    label_map = load_label_map(cfg, y_col) if cfg.TASK_MODE == "classification" else None

    n_rows = len(df)
    ranges = _chunk_ranges(n_rows, cfg.CHUNK_SIZE)

    base = _base_output(df, id_cols, y_actual, y_col)

    # which models to score
    best_path = cfg.TRAINED_MODELS_DIR / f"{y_col}_best.joblib"
    model_paths: Dict[str, str] = {}

    if cfg.PREDICTION_MODE == "best":
        if not best_path.exists():
            logger.warning("Missing best model for %s at %s", y_col, best_path)
            return None
        model_paths["BEST"] = str(best_path)
    else:
        paths = available_model_paths(cfg, y_col)
        if not paths and best_path.exists():
            model_paths["BEST"] = str(best_path)
        else:
            for k, v in paths.items():
                model_paths[k] = str(v)
        if not model_paths:
            logger.warning("No models found for %s; skipping.", y_col)
            return None

    out = base.copy()

    for model_key, path in model_paths.items():
        model = load(path)

        all_pred = []
        all_proba = []
        classes_from_model = None

        for (s, e) in ranges:
            idx = X_num.index[s:e]
            pred, proba, classes = _predict_chunk(model, X_num.loc[idx], X_cb.loc[idx])
            all_pred.append(np.asarray(pred))
            if proba is not None:
                all_proba.append(np.asarray(proba))
            if classes is not None:
                classes_from_model = classes

        pred_all = np.concatenate(all_pred, axis=0)

        # decode to original labels if map exists
        pred_label = decode_predictions(pred_all, label_map)

        # column names
        base_col = (
            f"{y_col}_interpolated_{cfg.MODEL_SUFFIX}"
            if model_key == "BEST"
            else f"{y_col}_{model_key}_interpolated_{cfg.MODEL_SUFFIX}"
        )

        out[f"{base_col}"] = pred_label

        # probability metric
        if all_proba:
            proba_all = np.concatenate(all_proba, axis=0)
            max_proba = proba_all.max(axis=1)
            out[f"{base_col}_metric1"] = max_proba

            if cfg.RETURN_CLASS_PROBABILITY_DICT:
                # class labels for dict: use label_map if available
                if label_map is not None:
                    cls_keys = [label_map.get(i, str(i)) for i in range(proba_all.shape[1])]
                elif classes_from_model is not None:
                    cls_keys = [str(c) for c in list(classes_from_model)]
                else:
                    cls_keys = [str(i) for i in range(proba_all.shape[1])]
                out[f"{base_col}_class_probability"] = proba_to_dict_list(cls_keys, proba_all)

        # global confidence metrics from Stage-2 results
        if cfg.RETURN_CONFIDENCE_METRICS and cfg.TASK_MODE == "classification":
            if cv_row is not None:
                # best model metrics when model_key == BEST, else use model_key
                mname = best_name if (model_key == "BEST" and best_name) else model_key
                if mname in {"RF", "LGBM", "XGB", "HGB", "CB"}:
                    prefix = f"{mname}_"
                    out[f"{base_col}_metric2_f1"] = cv_row.get(f"{prefix}f1_macro", np.nan)
                    out[f"{base_col}_metric3_recall"] = cv_row.get(f"{prefix}recall_macro", np.nan)
                    out[f"{base_col}_metric4_precision"] = cv_row.get(f"{prefix}precision_macro", np.nan)
                    out[f"{base_col}_metric5_auc"] = cv_row.get(f"{prefix}auc", np.nan)
                else:
                    out[f"{base_col}_metric2_f1"] = np.nan
                    out[f"{base_col}_metric3_recall"] = np.nan
                    out[f"{base_col}_metric4_precision"] = np.nan
                    out[f"{base_col}_metric5_auc"] = np.nan
            else:
                out[f"{base_col}_metric2_f1"] = np.nan
                out[f"{base_col}_metric3_recall"] = np.nan
                out[f"{base_col}_metric4_precision"] = np.nan
                out[f"{base_col}_metric5_auc"] = np.nan

    logger.info("Finished scoring for %s", y_col)
    return out


def postprocess_output(df: pd.DataFrame) -> pd.DataFrame:
    # keep same behavior you had
    if "id_review_id" in df.columns:
        df.insert(0, "snaap_sample", df["id_review_id"].notnull())
        df = df.drop(columns=["id_review_id"])
    df.columns = df.columns.str.replace(r"^(?:y_|id_)", "", regex=True)
    return df


def main() -> None:
    cfg = Stage3Config()
    LOG = init_root_logger()

    cfg.FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading scoring data from %s | DEV_ROW_SUBSET=%s", fetch_master_data_fn(), cfg.DEV_ROW_SUBSET)
    df = return_master_data(add_budget=True, model_type=cfg.MODEL_TYPE, scoring_or_training="scoring")

    if cfg.DEV_ROW_SUBSET is not None:
        if cfg.DEV_ROW_SUBSET_MODE == "sample":
            df = df.sample(n=min(cfg.DEV_ROW_SUBSET, len(df)), random_state=cfg.RANDOM_STATE).copy()
        else:
            df = df.head(cfg.DEV_ROW_SUBSET).copy()

    id_cols, ft_cols, _y_cols_from_data = detect_columns(df, cfg)

    df_cv = load_stage2_cv_results(cfg, LOG)

    # infer targets from saved models (preferred, matches your current logic)
    model_best_files = list(cfg.TRAINED_MODELS_DIR.glob("*_best.joblib"))
    targets_from_models = sorted({p.name.split("_best.joblib")[0] for p in model_best_files})

    if not targets_from_models:
        LOG.error("No *_best.joblib models found in %s; abort.", cfg.TRAINED_MODELS_DIR)
        return

    targets = cfg.filter_targets(targets_from_models)
    LOG.info("Targets to score (%d): %s", len(targets), targets)

    results = Parallel(n_jobs=cfg.N_JOBS_TARGETS)(
        delayed(predict_for_target)(t, df, id_cols, ft_cols, df_cv, cfg) for t in targets
    )

    frames = [r for r in results if r is not None]
    if not frames:
        LOG.error("No predictions generated.")
        return

    combined = frames[0]
    for fr in frames[1:]:
        combined = combined.join(fr.drop(columns=id_cols, errors="ignore"), how="outer")

    combined = postprocess_output(combined)

    checks_dict = validate_df(output=combined, master_data=df, MODEL_SUFFIX=cfg.MODEL_SUFFIX)
    with cfg.CHECKS_OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(checks_dict, f, indent=2, default=str, ensure_ascii=False)

    combined.to_csv(cfg.OUTPUT_PATH, index=False)
    LOG.info("Stage-3 predictions saved to %s", cfg.OUTPUT_PATH)

    # Optional: preserve your BM multi-select recombine behavior
    combined2 = recombine_multi_select(combined, cfg.MODEL_SUFFIX)
    combined2.to_csv(str(cfg.OUTPUT_PATH).replace(".csv", "_comb_multi.csv"), index=False)


if __name__ == "__main__":
    main()
