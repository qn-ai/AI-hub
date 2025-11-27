# data_preprocessing.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from config import RANDOM_STATE, TEST_SIZE


def load_data() -> pd.DataFrame:
    """
    Placeholder data loader.

    TODO: Replace with your real data loading logic.
    Currently:
      - creates synthetic features ft_*
      - synthetic multi-target columns y_main, y_alt1, y_alt2
      - injects some missing values.
    """
    from sklearn.datasets import make_classification

    X, y_main = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=5,
        random_state=RANDOM_STATE,
    )

    df = pd.DataFrame(X, columns=[f"ft_{i}" for i in range(X.shape[1])])
    df["y_main"] = y_main
    df["y_alt1"] = (y_main + 1) % 5
    df["y_alt2"] = (y_main + 2) % 5

    rng = np.random.default_rng(RANDOM_STATE)
    for col in df.columns:
        mask = rng.random(df.shape[0]) < 0.05
        df.loc[mask, col] = np.nan

    return df


def split_features_targets(
    df: pd.DataFrame,
    target_prefix: str = "y_",
    explicit_targets: List[str] | None = None,
    id_prefix: str = "id_",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Split dataframe into features X, target columns, and id columns.

    Targets:
      - if explicit_targets is given, use that
      - else, use all columns starting with target_prefix
        (with a fallback for synthetic y_main, y_alt1, y_alt2).

    Features:
      - all columns except targets and id_ columns.
    """
    if explicit_targets is not None:
        target_cols = explicit_targets
    else:
        target_cols = [c for c in df.columns if c.startswith(target_prefix)]
        if not target_cols:
            candidate = ["y_main", "y_alt1", "y_alt2"]
            target_cols = [c for c in candidate if c in df.columns]

    id_cols = [c for c in df.columns if c.startswith(id_prefix)]
    feature_cols = [c for c in df.columns if c not in target_cols + id_cols]

    X = df[feature_cols].copy()
    return X, target_cols, id_cols


def detect_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect numeric vs categorical feature columns.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_sklearn_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer for RF & LightGBM:
      - numeric: median imputation
      - categorical: most_frequent + OrdinalEncoder
    """
    numeric_cols, categorical_cols = detect_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ],
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
    )

    return preprocessor, numeric_cols, categorical_cols


def get_catboost_cat_indices(
    X: pd.DataFrame,
    categorical_cols: List[str],
) -> List[int]:
    """
    Return column indices for CatBoost categorical features.
    """
    return [X.columns.get_loc(c) for c in categorical_cols]


def train_val_split_for_target(
    df: pd.DataFrame,
    X: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder | None]:
    """
    For a given target:
      - Drop rows with missing target
      - Split into train/val
      - Label-encode target if necessary.
    """
    y = df[target_col]
    mask = y.notna()
    X_sub = X.loc[mask].copy()
    y_sub = y.loc[mask].copy()

    if y_sub.dtype == "object":
        le = LabelEncoder()
        y_sub = pd.Series(
            le.fit_transform(y_sub),
            index=y_sub.index,
            name=target_col,
        )
    else:
        le = None

    X_train, X_val, y_train, y_val = train_test_split(
        X_sub,
        y_sub,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_sub,
    )

    return X_train, X_val, y_train, y_val, le
