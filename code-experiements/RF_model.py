import warnings
warnings.filterwarnings("ignore")

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# =====================================================
DATA_PATH = "input_data.csv"
OUT_PATH = "output_interpolated_model1.csv"

MIN_LABELS = 200               # minimum labelled rows per target
IMBALANCE_THRESHOLD = 0.80     # for info only; not mandatory to use
FOLD_CANDIDATES = [3, 5, 10]   # candidate CV folds (optional – for logging)
N_JOBS = -1                    # parallelism for joblib
# =====================================================


def train_rf_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Instantiate and return a RandomForest model."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
        # you can add class_weight="balanced" if you want
    )
    return model


def process_target(df: pd.DataFrame,
                   target: str,
                   ft_cols: list[str]):
    """
    Train RF for a single target and return a DataFrame with:
      - y_<target>_interpolated_model1
      - y_<target>_interpolated_model1_confidence_metric1
    """
    print(f"\n=== Random Forest | Target {target} ===")

    # labelled rows only
    mask = df[target].notna()
    y = df.loc[mask, target]

    # skip if not enough labels or only one class
    if y.nunique() < 2 or len(y) < MIN_LABELS:
        print(f"  [SKIP] {target}: n_labelled={len(y)}, "
              f"n_classes={y.nunique()}")
        return None

    # features
    X_raw = df.loc[mask, ft_cols]
    X_full_raw = df[ft_cols]

    # one-hot encode all features
    X = pd.get_dummies(X_raw, dummy_na=True)
    X_full = pd.get_dummies(X_full_raw, dummy_na=True)
    X_full = X_full.reindex(columns=X.columns, fill_value=0)

    # optional: detect imbalance (for logging)
    class_dist = y.value_counts(normalize=True)
    maj_prop = class_dist.max()
    is_imbalanced = maj_prop >= IMBALANCE_THRESHOLD
    print(f"  Imbalanced: {is_imbalanced} "
          f"(majority={maj_prop:.3f})")

    # -------------------------------------------------
    # OPTIONAL: choose best CV folds (just for info)
    # -------------------------------------------------
    fold_scores = {}
    for k in FOLD_CANDIDATES:
        if len(y) < k:
            continue

        model_temp = train_rf_model(X, y)
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(
            model_temp, X, y, cv=cv,
            scoring="accuracy", n_jobs=-1
        )
        fold_scores[k] = (scores.mean(), scores.std())

    if fold_scores:
        best_k = max(fold_scores, key=lambda kk: fold_scores[kk][0])
        best_cv_mean, best_cv_std = fold_scores[best_k]
        print(f"  Best CV folds: {best_k} "
              f"(acc={best_cv_mean:.4f}, std={best_cv_std:.4f})")
    else:
        print("  Not enough samples for any CV folds.")

    # -------------------------------------------------
    # Train final RF on *all* labelled rows
    # -------------------------------------------------
    model = train_rf_model(X, y)

    t0 = time.time()
    model.fit(X, y)
    training_time = time.time() - t0
    print(f"  Training time: {training_time:.2f} sec")

    # basic train metrics (for monitoring, not used later)
    y_train_pred = model.predict(X)
    acc_train = accuracy_score(y, y_train_pred)
    f1_train = f1_score(y, y_train_pred, average="macro")
    print(f"  Train accuracy: {acc_train:.4f}, macro F1: {f1_train:.4f}")

    # -------------------------------------------------
    # Predict for ALL rows (including unlabelled)
    # -------------------------------------------------
    y_pred_all = model.predict(X_full)

    if hasattr(model, "predict_proba"):
        y_proba_all = model.predict_proba(X_full)

        # confidence = prob of positive class for binary,
        # otherwise the max class probability
        if y.nunique() == 2:
            conf_all = y_proba_all[:, 1]
        else:
            conf_all = y_proba_all.max(axis=1)
    else:
        conf_all = np.ones(len(X_full)) * np.nan

    base_name = f"{target}_interpolated_model1"
    preds_df = pd.DataFrame({
        base_name: y_pred_all,
        f"{base_name}_confidence_metric1": conf_all
    })

    return preds_df


def main():
    # -------------------------------------------------
    # Load data & identify columns
    # -------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    id_cols = [c for c in df.columns if c.startswith("id_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    ft_cols = [c for c in df.columns if c.startswith("ft_")]

    print(f"Loaded: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"ID cols: {id_cols}")
    print(f"Feature cols (ft_): {len(ft_cols)}")
    print(f"Target cols (y_): {len(y_cols)}")

    # start result table with IDs + original targets
    result_table = df[id_cols + y_cols].copy()

    # -------------------------------------------------
    # Process each target in parallel
    # -------------------------------------------------
    processed_list = Parallel(n_jobs=N_JOBS)(
        delayed(process_target)(df, target, ft_cols)
        for target in y_cols
    )

    # add prediction columns
    for preds_df in processed_list:
        if preds_df is None:
            continue
        result_table = pd.concat([result_table, preds_df], axis=1)

    # -------------------------------------------------
    # Save final table
    # -------------------------------------------------
    result_table.to_csv(OUT_PATH, index=False)
    print(f"\nSaved interpolated prediction table → {OUT_PATH}")
    print("DONE ✓")


if __name__ == "__main__":
    main()
