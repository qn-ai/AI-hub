from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.impute import SimpleImputer

from .stage3_config import Stage3Config


def init_root_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("stage3")


def get_target_logger(cfg: Stage3Config, y_col: str) -> logging.Logger:
    logger = logging.getLogger(f"stage3.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage3_file", False)
        for h in logger.handlers
    )
    if not exists:
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.LOG_DIR / f"{y_col}_stage3.log", mode="w", encoding="utf-8")
        fh._stage3_file = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.propagate = True
    return logger


def detect_columns(df: pd.DataFrame, cfg: Stage3Config) -> Tuple[List[str], List[str], List[str]]:
    if cfg.MODEL_TYPE == "predictassessment":
        id_cols = [c for c in df.columns if c.startswith(cfg.ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(cfg.FEATURE_PREFIX)]
        y_cols = [c for c in df.columns if c.startswith(cfg.TARGET_PREFIX)]
        return id_cols, ft_cols, y_cols

    if cfg.MODEL_TYPE == "predictbudget":
        id_cols = [c for c in df.columns if c.startswith(cfg.ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(cfg.FEATURE_PREFIX)]
        return id_cols, ft_cols, ["budget_total"]

    if cfg.MODEL_TYPE == "assessmentbudget":
        id_cols = [c for c in df.columns if c.startswith(cfg.ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(cfg.TARGET_PREFIX)]
        y_cols = [c for c in df.columns if c.startswith(cfg.BUDGET_PREFIX)]
        return id_cols, ft_cols, y_cols

    raise ValueError(f"Invalid MODEL_TYPE={cfg.MODEL_TYPE}")


def load_stage2_cv_results(cfg: Stage3Config, log: logging.Logger) -> Optional[pd.DataFrame]:
    if not cfg.CV_RESULTS_CSV.exists():
        log.warning("Stage-2 CV results not found at %s", cfg.CV_RESULTS_CSV)
        return None
    return pd.read_csv(cfg.CV_RESULTS_CSV)


def get_cv_row_for_target(df_cv: Optional[pd.DataFrame], y_col: str) -> Optional[pd.Series]:
    if df_cv is None:
        return None
    sub = df_cv[df_cv["target"] == y_col]
    if sub.empty:
        return None
    return sub.iloc[0]


def load_feature_importances_for_target(cfg: Stage3Config, y_col: str, logger: logging.Logger) -> Optional[List[str]]:
    path = cfg.FEATURE_IMPORTANCE_DIR / f"feature_importances_{y_col}.csv"
    if not path.exists():
        logger.warning("No feature_importances file for %s at %s", y_col, path)
        return None

    df_imp = pd.read_csv(path)
    if "feature_name" not in df_imp.columns or "mean_rank" not in df_imp.columns:
        logger.warning("feature_importances_%s.csv missing required columns; skipping.", y_col)
        return None

    selected = (
        df_imp[df_imp["mean_rank"] <= cfg.FEATURE_REDUCTION_TOP_N_FEATURES]["feature_name"]
        .astype(str)
        .tolist()
    )
    if not selected:
        logger.warning("No features selected for %s after importance filter; skipping.", y_col)
        return None
    return selected


def _coerce_object_cols_to_string(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    for c in X2.select_dtypes(include=["object"]).columns.tolist():
        X2[c] = X2[c].astype("string")
    return X2


def prepare_views_for_prediction(
    X: pd.DataFrame,
    y_optional: Optional[pd.Series],
    cfg: Stage3Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_num: numeric view used by RF/LGBM/XGB/HGB
      X_cb:  CatBoost view (string categoricals)
    Note:
      We fit CatBoostEncoder ONLY if y_optional is provided and has any non-null rows.
      Otherwise we drop object/string columns for X_num.
    """
    X_cb = _coerce_object_cols_to_string(X)
    for c in X_cb.select_dtypes(include=["string"]).columns.tolist():
        X_cb[c] = X_cb[c].fillna(cfg.CAT_FILL_VALUE)

    X_num = _coerce_object_cols_to_string(X)
    cat_cols = X_num.select_dtypes(include=["string"]).columns.tolist()

    if cfg.USE_CATBOOST_ENCODER and cat_cols and y_optional is not None:
        mask = y_optional.notna()
        if mask.any():
            enc = CatBoostEncoder(cols=cat_cols, random_state=cfg.RANDOM_STATE)
            enc.fit(X_num.loc[mask, :], y_optional.loc[mask])
            X_num = enc.transform(X_num)
        else:
            X_num = X_num.drop(columns=cat_cols)
    else:
        if cat_cols:
            X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    if cfg.NUM_IMPUTE:
        imp = SimpleImputer(strategy=cfg.NUM_IMPUTE)
        X_num = pd.DataFrame(imp.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

    return X_num, X_cb


def load_label_map(cfg: Stage3Config, y_col: str) -> Optional[Dict[int, str]]:
    path = cfg.TRAINED_MODELS_DIR / f"{y_col}_label_map.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # stored keys are strings
    return {int(k): str(v) for k, v in raw.items()}


def decode_predictions(pred_codes, label_map: Optional[Dict[int, str]]) -> List[str]:
    if label_map is None:
        # if model outputs strings already, just return as-is
        return [str(x) for x in list(pred_codes)]
    return [label_map.get(int(c), f"UNKNOWN_{c}") for c in list(pred_codes)]


def proba_to_dict_list(classes, proba: np.ndarray) -> List[Dict[str, str]]:
    cls = [str(c) for c in list(classes)]
    out: List[Dict[str, str]] = []
    for row in proba:
        out.append({c: f"{float(v):.4f}" for c, v in zip(cls, row)})
    return out


def available_model_paths(cfg: Stage3Config, y_col: str) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for name in cfg.ENABLED_MODELS:
        p = cfg.TRAINED_MODELS_DIR / f"{y_col}_{name}.joblib"
        if p.exists():
            paths[name] = p
    return paths
