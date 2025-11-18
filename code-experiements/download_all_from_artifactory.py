"""
Multi-task classification with missing labels per target.

- Assumes a CSV with:
    * Feature columns
    * Label columns starting with 'y_'
    * Optional ID column 'id_pwd_id'
- Trains one HistGradientBoostingClassifier per target (handles missing labels by
  training only on rows where that label is present).
- Evaluates on a validation split.
- Writes predictions back to a CSV alongside original data.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================
# CONFIG
# =========================
DATA_PATH = "input_data.csv"          # <-- change to your file path
OUTPUT_PATH = "input_with_predictions.csv"

TARGET_PREFIX = "y_"                  # label columns start with this
MIN_LABELLED = 500                    # minimum labelled samples per target
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    # -------------------------
    # 1. Load data
    # -------------------------
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    n_rows, n_cols = df.shape
    print(f"Data shape: {n_rows} rows x {n_cols} columns")

    # Optional ID column (will be kept & used only for merge / inspection)
    id_col = "id_pwd_id" if "id_pwd_id" in df.columns else None
    if id_col:
        print(f"Detected ID column: {id_col}")

    # -------------------------
    # 2. Identify label & feature columns
    # -------------------------
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]

    if not target_cols:
        raise ValueError(f"No columns start with prefix '{TARGET_PREFIX}'.")

    print(f"Detected {len(target_cols)} target columns with prefix '{TARGET_PREFIX}'.")

    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    # -------------------------
    # 3. Filter targets by minimum labelled samples
    # -------------------------
    label_counts = Y.notna().sum().to_frame("n_labelled")
    label_counts["total"] = len(df)
    label_counts["n_missing"] = label_counts["total"] - label_counts["n_labelled"]
    label_counts["frac_missing"] = label_counts["n_missing"] / label_counts["total"]

    print("\nLabelled count per target (head):")
    print(label_counts.sort_values("n_labelled").head(10))

    good_targets = [
        col for col in target_cols
        if Y[col].notna().sum() >= MIN_LABELLED
    ]
    bad_targets = sorted(set(target_cols) - set(good_targets))

    print(f"\nTargets kept (>= {MIN_LABELLED} labelled): {len(good_targets)}")
    print(f"Targets dropped (< {MIN_LABELLED} labelled): {len(bad_targets)}")
    if bad_targets:
        print("Example dropped targets:", bad_targets[:10])

    if not good_targets:
        raise ValueError(
            f"No targets have at least {MIN_LABELLED} labelled samples. "
            "Lower MIN_LABELLED and retry."
        )

    Y_good = Y[good_targets].copy()

    # -------------------------
    # 4. Train/validation split
    # -------------------------
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_good, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\nTrain size: {len(X_train)} rows")
    print(f"Validation size: {len(X_val)} rows")

    # -------------------------
    # 5. Train one model per target
    # -------------------------
    models = {}
    accs = {}

    print("\nTraining models per target...")
    for col in good_targets:
        y_train_col = Y_train[col]

        # Use only rows where this target is labelled
        mask_train = ~y_train_col.isna()
        n_train_labelled = mask_train.sum()

        # Skip if not enough labelled data in TRAIN split
        if n_train_labelled < max(100, MIN_LABELLED // 3):
            print(
                f"  [SKIP] {col}: only {n_train_labelled} labelled rows in train "
                f"(< {max(100, MIN_LABELLED // 3)})"
            )
            accs[col] = np.nan
            continue

        # Need at least 2 classes
        unique_classes = y_train_col[mask_train].unique()
        if len(unique_classes) < 2:
            print(
                f"  [SKIP] {col}: only one class present in train "
                f"({unique_classes})"
            )
            accs[col] = np.nan
            continue

        clf = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=None,
            max_bins=255,
            l2_regularization=0.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )

        clf.fit(X_train[mask_train], y_train_col[mask_train])
        models[col] = clf

        # Validation accuracy: only rows where target is labelled in val
        y_val_col = Y_val[col]
        mask_val = ~y_val_col.isna()

        if mask_val.sum() == 0:
            print(f"  [NO VAL LABELS] {col}: accuracy set to NaN")
            accs[col] = np.nan
            continue

        y_val_true = y_val_col[mask_val]
        y_val_pred = clf.predict(X_val[mask_val])

        acc = accuracy_score(y_val_true, y_val_pred)
        accs[col] = acc

        print(f"  [OK] {col}: train_labelled={n_train_labelled}, "
              f"val_labelled={mask_val.sum()}, accuracy={acc:.4f}")

    # -------------------------
    # 6. Accuracy summary: per target + min/max
    # -------------------------
    accs_series = pd.Series(accs, name="accuracy")
    accs_df = accs_series.to_frame().sort_values("accuracy")

    print("\nPer-target validation accuracy (sorted):")
    print(accs_df)

    # Exclude NaN when computing min/max
    valid_accs = accs_series.dropna()
    if not valid_accs.empty:
        min_acc = valid_accs.min()
        max_acc = valid_accs.max()
        print(f"\nMin validation accuracy: {min_acc:.4f}")
        print(f"Max validation accuracy: {max_acc:.4f}")
    else:
        print("\nNo valid accuracies (all NaN). Check your data/thresholds.")

    # Optionally save accuracies
    accs_df.to_csv("per_target_accuracy.csv", index=True)
    print("Per-target accuracies saved to: per_target_accuracy.csv")

    # -------------------------
    # 7. Predict for ALL rows and merge back into original df
    # -------------------------
    print("\nPredicting labels for ALL rows and merging with input data...")

    # We'll use the models trained on the train split (no refit on full data)
    # and predict for ALL X.
    preds = {}
    for col, clf in models.items():
        preds[col + "_pred"] = clf.predict(X)

    preds_df = pd.DataFrame(preds, index=df.index)

    # Merge predictions with original data
    df_with_preds = pd.concat([df, preds_df], axis=1)

    # Save to CSV
    df_with_preds.to_csv(OUTPUT_PATH, index=False)
    print(f"Full data with predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
