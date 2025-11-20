import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# =====================================================
DATA_PATH = "input_data.csv"
OUT_PATH = "output_interpolated_model1.csv"

MIN_LABELS = 200          # minimum labelled rows per target
ROW_CHUNK_SIZE = 50000    # how many rows per prediction batch
RANDOM_STATE = 42
# =====================================================


def train_rf_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Instantiate and train a RandomForest model."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        # class_weight="balanced"  # uncomment if you want imbalance handling
    )
    model.fit(X, y)
    return model


def predict_in_row_chunks(
    model: RandomForestClassifier,
    X_full: pd.DataFrame,
    is_binary: bool,
    row_chunk_size: int = ROW_CHUNK_SIZE
):
    """
    Predict labels + confidence in row chunks to reduce memory & avoid timeouts.

    Returns:
        y_pred_all: np.ndarray of shape (n_rows,)
        conf_all:   np.ndarray of shape (n_rows,)
    """
    n = len(X_full)
    all_pred = []
    all_conf = []

    for start in range(0, n, row_chunk_size):
        end = min(start + row_chunk_size, n)
        X_batch = X_full.iloc[start:end]

        y_pred_batch = model.predict(X_batch)
        proba_batch = model.predict_proba(X_batch)

        if is_binary:
            # prob of class 1
            conf_batch = proba_batch[:, 1]
        else:
            # max prob among classes
            conf_batch = proba_batch.max(axis=1)

        all_pred.append(y_pred_batch)
        all_conf.append(conf_batch)

        print(f"    Predicted rows {start}–{end-1}")

    y_pred_all = np.concatenate(all_pred)
    conf_all = np.concatenate(all_conf)
    return y_pred_all, conf_all


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
    print(f"# Feature cols (ft_): {len(ft_cols)}")
    print(f"# Target cols (y_): {len(y_cols)}")

    # -------------------------------------------------
    # Encode all features ONCE (for all targets)
    # -------------------------------------------------
    print("\nEncoding features with get_dummies (once)...")
    X_full_raw = df[ft_cols]
    X_full = pd.get_dummies(X_full_raw, dummy_na=True)
    print(f"  Encoded feature shape: {X_full.shape}")

    # -------------------------------------------------
    # Start result table with IDs + original targets
    # -------------------------------------------------
    result_table = df[id_cols + y_cols].copy()

    # -------------------------------------------------
    # Process each target sequentially
    # -------------------------------------------------
    for target in y_cols:
        print(f"\n=== Random Forest | Target {target} ===")

        mask = df[target].notna()
        y = df.loc[mask, target]

        # skip if not enough labels or only one class
        if y.nunique() < 2 or len(y) < MIN_LABELS:
            print(f"  [SKIP] {target}: n_labelled={len(y)}, "
                  f"n_classes={y.nunique()}")
            continue

        # labelled subset of encoded features
        X = X_full.loc[mask]
        print(f"  n_labelled: {len(y)}, n_classes: {y.nunique()}")

        # ---- train RF ----
        t0 = time.time()
        model = train_rf_model(X, y)
        training_time = time.time() - t0
        print(f"  Training time: {training_time:.2f} sec")

        # ---- predict for ALL rows in chunks ----
        is_binary = (y.nunique() == 2)
        print("  Predicting for all rows in chunks...")
        y_pred_all, conf_all = predict_in_row_chunks(
            model, X_full, is_binary, ROW_CHUNK_SIZE
        )

        base_name = f"{target}_interpolated_model1"
        preds_df = pd.DataFrame({
            base_name: y_pred_all,
            f"{base_name}_confidence_metric1": conf_all
        })

        # merge into result table (columns will be reordered later)
        result_table = pd.concat([result_table, preds_df], axis=1)

    # -------------------------------------------------
    # Reorder columns
    #   id_pwd_id first (if exists), then other id_ cols,
    #   then for each y_:
    #     y,
    #     y_interpolated_model1,
    #     y_interpolated_model1_confidence_metric1 (if they exist)
    # -------------------------------------------------
    col_order = []

    # 1) id_pwd_id first if present
    if "id_pwd_id" in result_table.columns:
        col_order.append("id_pwd_id")

    # 2) other id_ columns (excluding id_pwd_id if already added)
    for c in id_cols:
        if c == "id_pwd_id":
            continue
        if c in result_table.columns and c not in col_order:
            col_order.append(c)

    # 3) for each y_ column: y, y_interpolated_model1, y_interpolated_model1_confidence_metric1
    for y_col in y_cols:
        # original target
        if y_col in result_table.columns:
            col_order.append(y_col)

        base_name = f"{y_col}_interpolated_model1"
        conf_name = f"{base_name}_confidence_metric1"

        if base_name in result_table.columns:
            col_order.append(base_name)
        if conf_name in result_table.columns:
            col_order.append(conf_name)

    # Keep only columns that actually exist
    col_order = [c for c in col_order if c in result_table.columns]

    # Finally reorder
    result_table = result_table[col_order]

    # -------------------------------------------------
    # Save final table
    # -------------------------------------------------
    result_table.to_csv(OUT_PATH, index=False)
    print(f"\nSaved interpolated prediction table → {OUT_PATH}")
    print("DONE ✓")


if __name__ == "__main__":
    main()
