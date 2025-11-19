import warnings
warnings.filterwarnings("ignore")

import time

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =====================================================
DATA_PATH = "input_data.csv"

# which models to run
MODELS = {
    "RF": "Random Forest",
    "LGBM": "LightGBM",
    "CAT": "CatBoost"
}

MIN_LABELS = 200               # minimum labelled rows per target
IMBALANCE_THRESHOLD = 0.80     # e.g. 80% in one class = imbalanced
FOLD_CANDIDATES = [3, 5, 10]   # candidate CV folds
N_JOBS = -1                    # parallelism for joblib
SHAP_SAMPLE = 500              # max rows for SHAP summary
# =====================================================


def detect_feature_types(df: pd.DataFrame, feature_cols):
    """
    Just for information / logging: work out which ft_ columns are numeric vs categorical.
    This DOES NOT transform the data; encoding happens later via get_dummies.
    """
    num_cols = []
    cat_cols = []

    for c in feature_cols:
        series = df[c]

        if (pd.api.types.is_float_dtype(series)
                or pd.api.types.is_integer_dtype(series)
                or pd.api.types.is_bool_dtype(series)):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return num_cols, cat_cols


def train_model(model_key: str, X: pd.DataFrame, y: pd.Series):
    """Return an instantiated model for the given key."""
    if model_key == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

    if model_key == "LGBM":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )

    if model_key == "CAT":
        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss" if y.nunique() == 2 else "MultiClass",
            verbose=False,
            random_seed=42
        )

    raise ValueError(f"Unknown model_key: {model_key}")


def process_target(df: pd.DataFrame,
                   target: str,
                   ft_cols: list[str],
                   model_key: str):
    """
    Train + evaluate a single target column with a given model family.

    Returns:
        (metrics_row: dict, predictions_df: pd.DataFrame) or None if skipped.
    """
    model_name = MODELS[model_key]
    print(f"\n=== {model_name} | Target {target} ===")

    # use only rows where this target is labelled
    mask = df[target].notna()
    y = df.loc[mask, target]

    # if too few labels or only one class -> skip
    if y.nunique() < 2 or len(y) < MIN_LABELS:
        print(f"  [SKIP] target={target}: "
              f"n_labelled={len(y)}, n_classes={y.nunique()}")
        return None

    # raw feature subsets
    X_raw = df.loc[mask, ft_cols]
    X_full_raw = df[ft_cols]

    # -------------------------------------------------
    # Encode all ft_ columns with get_dummies so that:
    #  - RF / LGBM / CAT can all consume them
    #  - no string->float conversion errors
    # -------------------------------------------------
    X = pd.get_dummies(X_raw, dummy_na=True)
    X_full = pd.get_dummies(X_full_raw, dummy_na=True)

    # make sure X_full has the same columns/order as X
    X_full = X_full.reindex(columns=X.columns, fill_value=0)

    # -------------------------------------------------
    # Imbalance detection
    # -------------------------------------------------
    class_dist = y.value_counts(normalize=True)
    maj_prop = class_dist.max()
    is_imbalanced = maj_prop >= IMBALANCE_THRESHOLD

    # -------------------------------------------------
    # Choose best CV fold count
    # -------------------------------------------------
    fold_scores = {}
    for k in FOLD_CANDIDATES:
        if len(y) < k:
            continue  # cannot have more folds than samples

        model_temp = train_model(model_key, X, y)
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model_temp, X, y, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        fold_scores[k] = (scores.mean(), scores.std())

    if not fold_scores:
        print(f"  [SKIP] target={target}: not enough samples for any CV folds")
        return None

    best_k = max(fold_scores, key=lambda k: fold_scores[k][0])
    best_cv_mean, best_cv_std = fold_scores[best_k]
    print(f"  Best CV folds: {best_k}  "
          f"(acc={best_cv_mean:.4f}, std={best_cv_std:.4f})")

    # -------------------------------------------------
    # Train final model on all labelled rows
    # -------------------------------------------------
    model = train_model(model_key, X, y)

    t0 = time.time()
    model.fit(X, y)
    training_time = time.time() - t0
    print(f"  Training time: {training_time:.2f} sec")

    # -------------------------------------------------
    # In-sample metrics (for monitoring only)
    # -------------------------------------------------
    y_train_pred = model.predict(X)
    acc_train = accuracy_score(y, y_train_pred)
    f1_train = f1_score(y, y_train_pred, average="macro")
    print(f"  Train accuracy: {acc_train:.4f}, macro F1: {f1_train:.4f}")

    # -------------------------------------------------
    # Predictions for ALL rows (including originally unlabelled)
    # -------------------------------------------------
    y_pred_all = model.predict(X_full)

    if hasattr(model, "predict_proba"):
        y_proba_all = model.predict_proba(X_full)
        # confidence = prob of positive class for binary, else max prob
        if y.nunique() == 2:
            conf_all = y_proba_all[:, 1]
        else:
            conf_all = y_proba_all.max(axis=1)
    else:
        conf_all = np.ones(len(X_full)) * np.nan

    pred_cols = {
        f"{target}_{model_key}_pred": y_pred_all,
        f"{target}_{model_key}_conf": conf_all,
    }
    preds_df = pd.DataFrame(pred_cols)

    # -------------------------------------------------
    # SHAP summary plot (optional, try/except)
    # -------------------------------------------------
    shap_path = f"shap_{model_key}_{target}.png"
    try:
        sample_X = X.sample(min(len(X), SHAP_SAMPLE), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_X)

        plt.figure()
        if y.nunique() == 2:
            shap.summary_plot(shap_values, sample_X, show=False)
        else:
            # multiclass: show class 0 by default
            shap.summary_plot(shap_values[0], sample_X, show=False)

        plt.tight_layout()
        plt.savefig(shap_path, dpi=120)
        plt.close()
        print(f"  SHAP summary saved → {shap_path}")
    except Exception as e:
        print(f"  SHAP error: {e}")

    # -------------------------------------------------
    # Collect metrics
    # -------------------------------------------------
    metrics_row = {
        "target": target,
        "model": MODELS[model_key],
        "n_labelled": len(y),
        "n_classes": y.nunique(),
        "cv_mean_acc": best_cv_mean,
        "cv_std_acc": best_cv_std,
        "best_cv_folds": best_k,
        "accuracy_train": acc_train,
        "f1_train": f1_train,
        "is_imbalanced": is_imbalanced,
        "majority_class_prop": maj_prop,
        "training_time_sec": training_time,
    }

    return metrics_row, preds_df


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

    # optional: just log feature types
    print("\nDetecting feature types...")
    num_cols, cat_cols = detect_feature_types(df, ft_cols)
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features (incl. strings): {len(cat_cols)}")

    # -------------------------------------------------
    # Train each model family separately
    # -------------------------------------------------
    for model_key in MODELS.keys():

        print("\n======================================")
        print(f"   TRAINING MODEL FAMILY: {MODELS[model_key]}")
        print("======================================")

        metrics_all = []
        # start table with IDs + original targets
        result_table = df[id_cols + y_cols].copy()

        processed = Parallel(n_jobs=N_JOBS)(
            delayed(process_target)(df, target, ft_cols, model_key)
            for target in y_cols
        )

        for out in processed:
            if out is None:
                continue
            metrics_row, preds_df = out

            metrics_all.append(metrics_row)
            result_table = pd.concat([result_table, preds_df], axis=1)

        # save outputs
        out_table_path = f"output_table_{model_key}.csv"
        out_metrics_path = f"metrics_{model_key}.csv"

        result_table.to_csv(out_table_path, index=False)
        pd.DataFrame(metrics_all).to_csv(out_metrics_path, index=False)

        print(f"\nSaved predictions table: {out_table_path}")
        print(f"Saved metrics:           {out_metrics_path}")

    print("\nALL MODELS COMPLETE ✓")


if __name__ == "__main__":
    main()
