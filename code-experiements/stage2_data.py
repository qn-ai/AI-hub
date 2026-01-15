from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold

from .stage2_config import Stage2Config


# -----------------------------
# LOGGING
# -----------------------------
def init_root_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("stage2")


def get_target_logger(cfg: Stage2Config, y_col: str) -> logging.Logger:
    logger = logging.getLogger(f"stage2.{y_col}")
    logger.setLevel(logging.INFO)

    exists = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_stage2_file", False)
        for h in logger.handlers
    )
    if not exists:
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.LOG_DIR / f"{y_col}_stage2.log", mode="w", encoding="utf-8")
        fh._stage2_file = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.propagate = True
    return logger


# -----------------------------
# COLUMNS
# -----------------------------
def detect_columns(df: pd.DataFrame, cfg: Stage2Config) -> Tuple[List[str], List[str], List[str]]:
    if cfg.MODEL_TYPE == "predictassessment":
        id_cols = [c for c in df.columns if c.startswith(cfg.ID_PREFIX)]
        ft_cols = [c for c in df.columns if c.startswith(cfg.FEATURE_PREFIX)]
        # Default detected list; you can override via cfg.TARGETS_INCLUDE
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


# -----------------------------
# FEATURE SELECTION
# -----------------------------
def load_feature_importances_for_target(
    y_col: str,
    cfg: Stage2Config,
    logger: logging.Logger,
) -> Optional[List[str]]:
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

    logger.info("Selected %d features for %s from Stage-1 importances.", len(selected), y_col)
    return selected


# -----------------------------
# TARGET CLEANING (mixed values fix)
# -----------------------------
def clean_target_and_filter_rows(
    df: pd.DataFrame,
    y_col: str,
    cfg: Stage2Config,
) -> Tuple[pd.DataFrame, pd.Series]:
    df_target = df[df[y_col].notna()].copy()

    if cfg.TASK_MODE == "regression":
        y = pd.to_numeric(df_target[y_col], errors="coerce")
        good = y.notna()
        df_target = df_target.loc[good].copy()
        y = y.loc[good]
        return df_target, y

    # classification: stable string labels
    y = df_target[y_col].astype("string").str.strip()
    y = y.replace("", pd.NA)
    good = y.notna()
    df_target = df_target.loc[good].copy()
    y = y.loc[good]
    return df_target, y


# -----------------------------
# FEATURE VIEWS (mixed feature dtypes fix)
# -----------------------------
def _coerce_object_cols_to_string(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    obj_cols = X2.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        X2[c] = X2[c].astype("string")
    return X2


def prepare_views(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Stage2Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # CatBoost view: keep string categoricals
    X_cb = _coerce_object_cols_to_string(X)
    cb_cat_cols = X_cb.select_dtypes(include=["string"]).columns.tolist()
    for c in cb_cat_cols:
        X_cb[c] = X_cb[c].fillna(cfg.CAT_FILL_VALUE)

    # Numeric view: encode categoricals then coerce to numeric
    X_num = _coerce_object_cols_to_string(X)
    cat_cols = X_num.select_dtypes(include=["string"]).columns.tolist()

    if cfg.USE_CATBOOST_ENCODER and cat_cols:
        enc = CatBoostEncoder(cols=cat_cols, random_state=cfg.RANDOM_STATE)
        X_num = enc.fit_transform(X_num, y)
    elif cat_cols:
        X_num = X_num.drop(columns=cat_cols)

    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    if cfg.NUM_IMPUTE:
        imp = SimpleImputer(strategy=cfg.NUM_IMPUTE)
        X_num = pd.DataFrame(imp.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

    return X_num, X_cb


# -----------------------------
# CV chooser
# -----------------------------
def choose_cv(y: pd.Series, cfg: Stage2Config, logger: logging.Logger):
    if cfg.TASK_MODE == "regression":
        return KFold(
            n_splits=cfg.MAX_N_SPLITS_REGRESSION,
            shuffle=True,
            random_state=cfg.RANDOM_STATE,
        )

    counts = y.value_counts()
    min_count = int(counts.min())
    n_splits = min(cfg.MAX_N_SPLITS_CLASSIFICATION, min_count)

    logger.info("Class distribution=%s; chosen n_splits=%d", counts.to_dict(), n_splits)

    if n_splits < 2:
        logger.warning("Cannot build StratifiedKFold: min_class_count=%d", min_count)
        return None

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
