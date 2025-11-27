# data_preprocessing.py
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE, TEST_SIZE


def load_data():
    """
    TODO: Replace this with your real data loading.
    For now: synthetic multiclass + multiple targets.
    """
    from sklearn.datasets import make_classification

    # Example: base features
    X, y_main = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=5,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f"ft_{i}" for i in range(X.shape[1])])

    # Add synthetic extra target columns (multi-target)
    df["y_main"] = y_main
    df["y_alt1"] = (y_main + 1) % 5  # just to simulate multiple targets
    df["y_alt2"] = (y_main + 2) % 5

    # Introduce some missing values randomly (features + targets)
    rng = np.random.default_rng(RANDOM_STATE)
    for col in df.columns:
        mask = rng.random(df.shape[0]) < 0.05  # 5% missing
        df.loc[mask, col] = np.nan

    return df


def split_features_targets(df, target_prefix="y_", explicit_targets=None, id_prefix="id_"):
    """
    Split dataframe into features X and list of target columns.
    - Targets: either all columns starting with target_prefix OR explicit list.
    - Features: everything else except id_ columns and targets.
    """
    if explicit_targets is not None:
        target_cols = explicit_targets
    else:
        target_cols = [c for c in df.columns if c.startswith(target_prefix)]
        if not target_cols:
            # Fallback: use example targets
            target_cols = [c for c in df.columns if c.startswith("y_")] or ["y_main", "y_alt1", "y_alt2"]

    id_cols = [c for c in df.columns if c.startswith(id_prefix)]
    feature_cols = [c for c in df.columns if c not in target_cols + id_cols]

    X = df[feature_cols].copy()
    return X, target_cols, id_cols


def detect_feature_types(X: pd.DataFrame):
    """
    Detect numeric vs categorical columns.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_sklearn_preprocessor(X: pd.DataFrame):
    """
    Build ColumnTransformer for RF & LightGBM:
    - numeric: median imputation
    - categorical: most_frequent + OrdinalEncoder
    """
    numeric_cols, categorical_cols = detect_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def get_catboost_cat_indices(X: pd.DataFrame, categorical_cols):
    """
    CatBoost wants column indices for categorical features.
    We assume X is a DataFrame.
    """
    return [X.columns.get_loc(c) for c in categorical_cols]


def train_val_split_for_target(df, X, target_col):
    """
    For a given target:
    - Drop rows with missing target
    - Split into train/val using consistent random_state.
    - Encode target if needed.
    """
    y = df[target_col]

    # Drop rows where target is NaN (we can't train without labels)
    mask = y.notna()
    X_sub = X.loc[mask].copy()
    y_sub = y.loc[mask].copy()

    # Encode label if not numeric
    if y_sub.dtype == "object":
        le = LabelEncoder()
        y_sub = pd.Series(le.fit_transform(y_sub), index=y_sub.index, name=target_col)
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
