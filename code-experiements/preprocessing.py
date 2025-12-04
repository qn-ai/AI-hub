# preprocessing.py
from typing import List

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE


def build_numeric_pipeline(
    base_estimator,
    feature_names: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    """Pipeline for RF/LGBM/XGB:
    - Impute numeric columns with median.
    - Target-encode categorical columns (CatBoostEncoder).
    """
    df_dummy = pd.DataFrame(columns=feature_names)
    num_cols = df_dummy.select_dtypes(include=[np.number]).columns.tolist()
    # In practice, we infer dtypes from actual X inside CV, but here we just
    # separate using provided categorical list.
    num_cols = [c for c in feature_names if c not in categorical_cols]

    num_transformer = SimpleImputer(strategy="median")

    cat_transformer = CatBoostEncoder(cols=categorical_cols, random_state=RANDOM_STATE)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", base_estimator),
        ]
    )
    return pipe
