# analyze_erccfs_multiclass_multioutput.py
# ERCCFS pipeline for multi-output multiclass data (32 features, 108 label columns).
# - Combines label pairs with mode (keeps integer classes)
# - Computes label correlation matrix P
# - Runs ERCCFS (plug in real Eq.15/17/19 in erccfs.py for best results)
# - Visualizes feature–label correlations and |W|
# - Evaluates with multi-output multiclass metrics:
#     * Hamming loss (lower better)
#     * Mean per-output accuracy
#     * Mean per-output macro-F1

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from erccfs import make_placeholder_erccfs  # Replace placeholders with Eqs. (15)(17)(19) for true ERCCFS

# ---------------- Step 1: Load data ----------------
def load_data(csv_path, feature_prefix="f", label_prefix="y"):
    df = pd.read_csv(csv_path)
    X_df = df[[c for c in df.columns if c.startswith(feature_prefix)]]
    Y_df = df[[c for c in df.columns if c.startswith(label_prefix)]]
    X = X_df.to_numpy(dtype=float)
    # IMPORTANT: Keep labels as integers; do NOT average or coerce to float classes
    Y = Y_df.to_numpy()
    if not np.issubdtype(Y.dtype, np.integer):
        # if file parsed them as floats, but they represent ints, cast safely
        if np.all(np.isfinite(Y)) and np.all(np.mod(Y, 1) == 0):
            Y = Y.astype(int)
        else:
            raise ValueError("Label matrix Y must be integer-encoded classes (e.g., 0..7 / 0..8).")
    if X.shape[1] != 32:
        print(f"[warn] Expected 32 features, found {X.shape[1]}. Proceeding.")
    if Y.shape[1] != 108:
        print(f"[warn] Expected 108 label columns, found {Y.shape[1]}. Proceeding.")
    return X, Y, X_df.columns.tolist(), Y_df.columns.tolist()

# ---------------- Step 2: Combine label pairs (keep integer classes) ----------------
def combine_label_pairs(Y, label_names):
    """
    Combine every two label columns into one integer class via mode.
    Example: 108 columns -> 54 outputs. Handles mixed ranges (0..7 and 0..8).
    """
    if Y.shape[1] % 2 != 0:
        raise ValueError("Label columns must be even to form pairs (e.g., 108).")
    combined = []
    names = []
    for i in range(0, Y.shape[1], 2):
        pair = Y[:, i:i+2]  # shape (n, 2)
        # mode across the two sources for the same concept
        m = mode(pair, axis=1, keepdims=False).mode  # shape (n,)
        combined.append(m)
        names.append(f"group_{i//2 + 1}")
    Y_reduced = np.column_stack(combined).astype(int)  # (n, 54)
    # Sanity: ensure integers and reasonable ranges
    # (some outputs may be 0..7, others 0..8; we do not remap)
    return Y_reduced, names

# ---------------- Step 3: Compute label correlation matrix ----------------
def compute_label_corr(Y_reduced):
    # Correlation across outputs; integers are fine here
    P = np.corrcoef(Y_reduced.T)
    # NaNs can happen on constant outputs; fill diagonal and nan with 0 except diag
    P = np.nan_to_num(P, nan=0.0)
    np.fill_diagonal(P, 1.0)
    return P

# ---------------- Step 4: Run ERCCFS ----------------
def run_erccfs(X, Y, P, n_features=15, max_outer=10, max_inner=100, tol_inner=1e-5):
    model = make_placeholder_erccfs(
        n_features=n_features,
        stability_overlap=0.85,
        max_outer_iters=max_outer,
        max_inner_iters=max_inner,
        tol_inner=tol_inner,
    )
    model.fit(X, Y, user_ctx={"P": P})
    return model

# ---------------- Step 5: Visualizations ----------------
def visualize_feature_label_corr(X, Y_reduced, feature_names, label_names):
    corr = np.corrcoef(np.hstack([X, Y_reduced]).T)
    corr_block = corr[:X.shape[1], X.shape[1]:]
    plt.figure(figsize=(12, 6))
    plt.imshow(corr_block, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Feature–Label Correlation Heatmap")
    plt.xlabel("Reduced Labels (54)")
    plt.ylabel("Features (32)")
    plt.tight_layout()
    plt.savefig("feature_label_correlation.png", dpi=200)
    plt.close()

def visualize_W_matrix(W, feature_names, label_names):
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(W), aspect="auto")
    plt.colorbar(label="|W|")
    plt.title("ERCCFS |W| Heatmap (Feature Importance per Reduced Label)")
    plt.xlabel("Reduced Labels (54)")
    plt.ylabel("Features (32)")
    plt.tight_layout()
    plt.savefig("W_feature_label_heatmap.png", dpi=200)
    plt.close()

# ---------------- Step 6: Multi-output multiclass evaluation ----------------
def _make_catboost_multiclass(iterations=500, depth=6, learning_rate=0.1, random_state=42, verbose=False):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None
    # CatBoost multiclass (per output) inside MultiOutputClassifier
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_state,
        verbose=verbose,
        allow_writing_files=False
    )

def fit_predict_multioutput(estimator, X_train, Y_train, X_test):
    """
    estimator: base classifier that supports multiclass (e.g., CatBoostClassifier or RandomForestClassifier).
    Wrapped with MultiOutputClassifier to handle multiple outputs.
    Returns integer predictions of shape (n_samples, n_outputs).
    """
    clf = MultiOutputClassifier(estimator)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)  # integer class per output
    return Y_pred

def evaluate_multioutput_multiclass(Y_true, Y_pred):
    """
    Compute:
      - Hamming loss
      - mean per-output accuracy
      - mean per-output macro-F1
    """
    hl = hamming_loss(Y_true, Y_pred)

    # Per-output accuracy and macro-F1
    accs = []
    f1s = []
    n_out = Y_true.shape[1]
    for j in range(n_out):
        accs.append(accuracy_score(Y_true[:, j], Y_pred[:, j]))
        # macro-F1 across classes present in this output
        f1s.append(f1_score(Y_true[:, j], Y_pred[:, j], average="macro", zero_division=0))
    metrics = {
        "hamming_loss": float(hl),
        "mean_accuracy_per_output": float(np.mean(accs)),
        "mean_macro_f1_per_output": float(np.mean(f1s)),
    }
    return metrics

def evaluate_selected_vs_all(X_train, X_test, Y_train, Y_test, selected_idx,
                             use_catboost=True, cb_iterations=500, cb_depth=6, cb_lr=0.1, cb_verbose=False, seed=42):
    # Build estimator(s)
    estimator_sel = None
    if use_catboost:
        estimator_sel = _make_catboost_multiclass(
            iterations=cb_iterations, depth=cb_depth, learning_rate=cb_lr, random_state=seed, verbose=cb_verbose
        )
    if estimator_sel is None:
        print("[info] CatBoost not available; falling back to RandomForest.")
        estimator_sel = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)

    # Selected features
    Xtr_sel, Xte_sel = X_train[:, selected_idx], X_test[:, selected_idx]
    Y_pred_sel = fit_predict_multioutput(estimator_sel, Xtr_sel, Y_train, Xte_sel)
    metrics_sel = evaluate_multioutput_multiclass(Y_test, Y_pred_sel)

    # All features (fresh estimator for fairness)
    if isinstance(estimator_sel, RandomForestClassifier):
        estimator_all = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    else:
        estimator_all = _make_catboost_multiclass(
            iterations=cb_iterations, depth=cb_depth, learning_rate=cb_lr, random_state=seed, verbose=cb_verbose
        ) or RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)

    Y_pred_all = fit_predict_multioutput(estimator_all, X_train, Y_train, X_test)
    metrics_all = evaluate_multioutput_multiclass(Y_test, Y_pred_all)

    return metrics_sel, metrics_all

# ---------------- Step 7: Main pipeline ----------------
def main(csv_path, feature_prefix="f", label_prefix="y", k=10,
         use_catboost=True, cb_iterations=500, cb_depth=6, cb_lr=0.1, cb_verbose=False, seed=42):
    print("Loading data...")
    X, Y, feature_names, label_names = load_data(csv_path, feature_prefix, label_prefix)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} labels")

    print("Combining label pairs via mode (keeps integer classes)...")
    Y_reduced, combined_label_names = combine_label_pairs(Y, label_names)
    # Optional sanity print
    # for j in range(min(5, Y_reduced.shape[1])):
    #     vals = np.unique(Y_reduced[:, j])
    #     print(f"Output {j} unique classes: {vals}")

    print("Computing label correlation matrix P...")
    P = compute_label_corr(Y_reduced)

    print("Train/test split...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_reduced, test_size=0.3, random_state=seed, stratify=None)
    # NOTE: stratify=None because multi-output stratification is non-trivial. Consider custom split if needed.

    print(f"Running ERCCFS (k={k})...")
    model = run_erccfs(X_train, Y_train, P, n_features=k)

    print("Visualizing correlations...")
    visualize_feature_label_corr(X, Y_reduced, feature_names, combined_label_names)
    visualize_W_matrix(model.W_, feature_names, combined_label_names)

    print("Evaluating (Selected vs All) with", "CatBoost" if use_catboost else "RandomForest", "...")
    metrics_sel, metrics_all = evaluate_selected_vs_all(
        X_train, X_test, Y_train, Y_test, model.selected_features_,
        use_catboost=use_catboost,
        cb_iterations=cb_iterations, cb_depth=cb_depth, cb_lr=cb_lr, cb_verbose=cb_verbose, seed=seed
    )

    print("\nSelected feature names:", [feature_names[i] for i in model.selected_features_])

    summary = {
        "k": int(k),
        "selected_indices": list(map(int, model.selected_features_)),
        "selected_feature_names": [feature_names[i] for i in model.selected_features_],
        "metrics_selected": metrics_sel,
        "metrics_all": metrics_all,
        "catboost_params": {
            "used": bool(use_catboost),
            "iterations": cb_iterations,
            "depth": cb_depth,
            "learning_rate": cb_lr
        }
    }
    with open("erccfs_multiclass_multioutput_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTS ===")
    print("Selected features metrics:", json.dumps(metrics_sel, indent=2))
    print("All features metrics:", json.dumps(metrics_all, indent=2))
    print("\nSaved:")
    print(" - feature_label_correlation.png")
    print(" - W_feature_label_heatmap.png")
    print(" - erccfs_multiclass_multioutput_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ERCCFS multi-output multiclass analysis (32 features, 108 labels)")
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--feature-prefix", default="f", help="Prefix for feature columns")
    ap.add_argument("--label-prefix", default="y", help="Prefix for label columns")
    ap.add_argument("--k", type=int, default=10, help="Top-k features to select")
    ap.add_argument("--no-catboost", action="store_true", help="Disable CatBoost and use RandomForest fallback")
    ap.add_argument("--cb-iterations", type=int, default=500, help="CatBoost iterations")
    ap.add_argument("--cb-depth", type=int, default=6, help="CatBoost depth")
    ap.add_argument("--cb-lr", type=float, default=0.1, help="CatBoost learning rate")
    ap.add_argument("--cb-verbose", action="store_true", help="CatBoost verbose training")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    main(
        args.csv,
        feature_prefix=args.feature_prefix,
        label_prefix=args.label_prefix,
        k=args.k,
        use_catboost=not args.no_catboost,
        cb_iterations=args.cb_iterations,
        cb_depth=args.cb_depth,
        cb_lr=args.cb_lr,
        cb_verbose=args.cb_verbose,
        seed=args.seed
    )
