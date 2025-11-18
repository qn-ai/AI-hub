"""
Multi-task classification with:
- Probabilities
- Feature importances
- Cross-validation
- Categorical feature handling
- Parallelization

Conventions:
- Feature columns start with: ft_
- Target/label columns start with: y_
- ID columns start with: id_
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# =========================
# CONFIG
# =========================
DATA_PATH = "input_data.csv"              # <-- change to your file
OUTPUT_PATH = "input_with_predictions.csv"

FEATURE_PREFIX = "ft_"
TARGET_PREFIX = "y_"
ID_PREFIX = "id_"

MIN_LABELLED = 500                        # min labelled rows per target
CV_FOLDS = 5
RANDOM_STATE = 42

# RandomForest settings
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


def build_preprocessor(df, feature_cols):
    """Build ColumnTransformer with numeric + categorical pipelines."""
    num_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor, num_cols, cat_cols


def get_feature_names(preprocessor, num_cols, cat_cols):
    """Retrieve final feature names after preprocessing."""
    feature_names = []

    if num_cols:
        feature_names.extend(num_cols)

    if cat_cols:
        cat_transformer = preprocessor.named_transformers_.get("cat")
        if cat_transformer is not None:
            ohe = cat_transformer.named_steps["onehot"]
            cat_feat_names = ohe.get_feature_names_out(cat_cols)
            feature_names.extend(cat_feat_names)

    return np.array(feature_names)


def main():
    # -------------------------
    # 1. Load data
    # -------------------------
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    n_rows = len(df)
    print(f"Data shape: {n_rows} rows x {df.shape[1]} cols")

    # -------------------------
    # 2. Identify columns
    # -------------------------
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]

    print(f"ID columns: {id_cols}")
    print(f"Feature columns (ft_): {len(feature_cols)}")
    print(f"Target columns (y_): {len(target_cols)}")

    if not feature_cols:
        raise ValueError(f"No feature columns starting with '{FEATURE_PREFIX}'")
    if not target_cols:
        raise ValueError(f"No target columns starting with '{TARGET_PREFIX}'")

    X_full = df[feature_cols].copy()
    Y_full = df[target_cols].copy()

    # -------------------------
    # 3. Build preprocessor
    # -------------------------
    print("\nBuilding preprocessor...")
    preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_cols)

    # -------------------------
    # 4. Filter targets by number of labelled rows
    # -------------------------
    label_counts = Y_full.notna().sum().to_frame("n_labelled")
    label_counts["total"] = len(df)
    label_counts["n_missing"] = label_counts["total"] - label_counts["n_labelled"]
    label_counts["frac_missing"] = label_counts["n_missing"] / label_counts["total"]

    print("\nTargets sorted by labelled count (lowest first):")
    print(label_counts.sort_values("n_labelled").head(10))

    good_targets = [
        c for c in target_cols
        if Y_full[c].notna().sum() >= MIN_LABELLED
    ]
    bad_targets = sorted(set(target_cols) - set(good_targets))

    print(f"\nTargets kept (>= {MIN_LABELLED} labelled): {len(good_targets)}")
    print(f"Targets dropped (< {MIN_LABELLED} labelled): {len(bad_targets)}")
    if bad_targets:
        print("Example dropped targets:", bad_targets[:10])

    if not good_targets:
        raise ValueError("No targets meet MIN_LABELLED. Reduce threshold and retry.")

    # -------------------------
    # 5. Train per target
    # -------------------------
    per_target_results = []

    # We'll build these DataFrames column by column (avoids length issues)
    preds_labels_df = pd.DataFrame(index=df.index)
    preds_probs_df = pd.DataFrame(index=df.index)

    print("\nTraining models per target with CV...")
    for col in good_targets:
        print(f"\n=== Target: {col} ===")

        y = Y_full[col]
        mask = ~y.isna()          # rows where label is present
        n_labelled = mask.sum()

        print(f"  Labelled rows: {n_labelled}")

        if n_labelled < MIN_LABELLED:
            print(f"  [SKIP] Not enough labelled rows after filter.")
            per_target_results.append({
                "target": col,
                "n_labelled": n_labelled,
                "cv_mean_acc": np.nan,
                "cv_std_acc": np.nan,
                "train_acc_full": np.nan,
            })
            continue

        X_lab = X_full.loc[mask]
        y_lab = y.loc[mask]

        unique_classes = np.unique(y_lab)
        if len(unique_classes) < 2:
            print(f"  [SKIP] Only one class present: {unique_classes}")
            per_target_results.append({
                "target": col,
                "n_labelled": n_labelled,
                "cv_mean_acc": np.nan,
                "cv_std_acc": np.nan,
                "train_acc_full": np.nan,
            })
            continue

        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(**RF_PARAMS)),
        ])

        # 5a. Cross-validation accuracy
        print(f"  Running {CV_FOLDS}-fold CV...")
        cv_scores = cross_val_score(
            pipe,
            X_lab,
            y_lab,
            cv=CV_FOLDS,
            scoring="accuracy",
            n_jobs=-1,
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"  CV accuracy: mean={cv_mean:.4f}, std={cv_std:.4f}")

        # 5b. Fit on all labelled data
        print("  Fitting model on all labelled rows...")
        pipe.fit(X_lab, y_lab)

        y_lab_pred = pipe.predict(X_lab)
        train_acc_full = accuracy_score(y_lab, y_lab_pred)
        print(f"  Train accuracy on labelled rows: {train_acc_full:.4f}")

        # 5c. Feature importances
        clf = pipe.named_steps["clf"]
        pre_fitted = pipe.named_steps["pre"]

        feature_names = get_feature_names(pre_fitted, num_cols, cat_cols)
        importances = clf.feature_importances_

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        fi_path = f"feature_importances_{col}.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"  Feature importances saved → {fi_path}")

        # 5d. Predict for ALL rows (labels + probs) SAFELY
        print("  Predicting for ALL rows...")
        y_all_pred = pipe.predict(X_full)

        # Sanity check
        if len(y_all_pred) != n_rows:
            raise RuntimeError(
                f"Length mismatch for {col}: "
                f"pred={len(y_all_pred)}, expected={n_rows}"
            )

        preds_labels_df[col + "_pred"] = pd.Series(
            y_all_pred,
            index=df.index
        )

        # Predict probabilities if available
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba_all = pipe.predict_proba(X_full)
            if proba_all.shape[0] != n_rows:
                raise RuntimeError(
                    f"Proba length mismatch for {col}: "
                    f"proba={proba_all.shape[0]}, expected={n_rows}"
                )

            classes = pipe.named_steps["clf"].classes_

            if len(classes) == 2:
                pos_class = classes[1]
                prob_col_name = f"{col}_proba_{pos_class}"
                preds_probs_df[prob_col_name] = pd.Series(
                    proba_all[:, 1],
                    index=df.index
                )
            else:
                for i, cls in enumerate(classes):
                    prob_col_name = f"{col}_proba_{cls}"
                    preds_probs_df[prob_col_name] = pd.Series(
                        proba_all[:, i],
                        index=df.index
                    )

        per_target_results.append({
            "target": col,
            "n_labelled": n_labelled,
            "cv_mean_acc": cv_mean,
            "cv_std_acc": cv_std,
            "train_acc_full": train_acc_full,
        })

    # -------------------------
    # 6. Save per-target results
    # -------------------------
    results_df = pd.DataFrame(per_target_results)
    results_df = results_df.sort_values("cv_mean_acc", ascending=True)
    results_df.to_csv("per_target_results_cv.csv", index=False)
    print("\nPer-target CV results saved → per_target_results_cv.csv")

    valid = results_df["cv_mean_acc"].dropna()
    if not valid.empty:
        print(f"\nMin CV accuracy: {valid.min():.4f}")
        print(f"Max CV accuracy: {valid.max():.4f}")
    else:
        print("\nNo valid CV accuracies (all NaN).")

    # -------------------------
    # 7. Merge predictions & probabilities back to df
    # -------------------------
    print("\nMerging predictions and probabilities back to original dataframe...")

    # preds_probs_df might be empty if all models lacked predict_proba
    parts = [df, preds_labels_df]
    if not preds_probs_df.empty:
        parts.append(preds_probs_df)

    df_out = pd.concat(parts, axis=1)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"All predictions saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
