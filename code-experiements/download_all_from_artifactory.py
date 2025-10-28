# analyze_erccfs_multilabel.py
# ERCCFS pipeline for multi-label data (32 features, 108 labels)
# Combines correlated label pairs, computes label correlation matrix,
# runs ERCCFS, visualizes importance, and evaluates using multi-label metrics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, average_precision_score, coverage_error
from erccfs import make_placeholder_erccfs  # Uses your ERCCFS skeleton

# ---------------- Step 1: Load data ----------------
def load_data(csv_path, feature_prefix="f", label_prefix="y"):
    df = pd.read_csv(csv_path)
    X_df = df[[c for c in df.columns if c.startswith(feature_prefix)]]
    Y_df = df[[c for c in df.columns if c.startswith(label_prefix)]]
    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    return X, Y, X_df.columns.tolist(), Y_df.columns.tolist()

# ---------------- Step 2: Combine label pairs ----------------
def combine_label_pairs(Y, label_names):
    """
    Combine correlated label pairs (assuming labels are grouped in 2s).
    Example: 108 columns -> 54 combined labels.
    """
    combined = []
    combined_names = []
    for i in range(0, Y.shape[1], 2):
        avg = Y[:, i:i+2].mean(axis=1)
        combined.append(avg)
        combined_names.append(f"group_{i//2+1}")
    Y_reduced = np.column_stack(combined)
    return Y_reduced, combined_names

# ---------------- Step 3: Compute label correlation matrix ----------------
def compute_label_corr(Y_reduced):
    P = np.corrcoef(Y_reduced.T)
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
    plt.imshow(corr_block, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Feature–Label Correlation Heatmap")
    plt.xlabel("Reduced Labels")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("feature_label_correlation.png", dpi=200)
    plt.close()

def visualize_W_matrix(W, feature_names, label_names):
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(W), aspect="auto", cmap="viridis")
    plt.colorbar(label="|W|")
    plt.title("ERCCFS Weight Matrix (Feature Importance per Label)")
    plt.xlabel("Labels")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("W_feature_label_heatmap.png", dpi=200)
    plt.close()

# ---------------- Step 6: Evaluate ----------------
def evaluate_multilabel(X_train, X_test, Y_train, Y_test, model):
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier

    selected_idx = model.selected_features_
    Xtr_sel, Xte_sel = X_train[:, selected_idx], X_test[:, selected_idx]

    base_clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf = MultiOutputClassifier(base_clf)
    clf.fit(Xtr_sel, Y_train)
    Y_pred = np.clip(clf.predict_proba(Xte_sel), 0, 1)
    # Some sklearn versions return list of arrays
    if isinstance(Y_pred, list):
        Y_pred = np.column_stack([y[:, 1] if y.shape[1] > 1 else y[:, 0] for y in Y_pred])

    hl = hamming_loss(Y_test, (Y_pred > 0.5).astype(int))
    map_score = average_precision_score(Y_test, Y_pred)
    cov = coverage_error(Y_test, Y_pred)

    print(f"\nHamming Loss: {hl:.4f}")
    print(f"Mean Average Precision: {map_score:.4f}")
    print(f"Coverage Error: {cov:.4f}")
    return hl, map_score, cov

# ---------------- Step 7: Main pipeline ----------------
def main(csv_path, feature_prefix="f", label_prefix="y"):
    print("Loading data...")
    X, Y, feature_names, label_names = load_data(csv_path, feature_prefix, label_prefix)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} labels")

    print("Combining label pairs...")
    Y_reduced, combined_label_names = combine_label_pairs(Y, label_names)
    print(f"Reduced labels: {Y.shape[1]} → {Y_reduced.shape[1]}")

    print("Computing label correlation matrix P...")
    P = compute_label_corr(Y_reduced)

    print("Train/test split...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_reduced, test_size=0.3, random_state=42)

    print("Running ERCCFS...")
    model = run_erccfs(X_train, Y_train, P, n_features=10)

    print("Visualizing correlations...")
    visualize_feature_label_corr(X, Y_reduced, feature_names, combined_label_names)
    visualize_W_matrix(model.W_, feature_names, combined_label_names)

    print("Evaluating multi-label performance...")
    hl, map_score, cov = evaluate_multilabel(X_train, X_test, Y_train, Y_test, model)

    print("\nTop selected features:")
    top_features = [feature_names[i] for i in model.selected_features_]
    print(top_features)

    # Save summary
    summary = {
        "selected_features": top_features,
        "hamming_loss": hl,
        "mean_average_precision": map_score,
        "coverage_error": cov,
        "n_features": X.shape[1],
        "n_selected": len(top_features),
    }
    with open("erccfs_multilabel_summary.json", "w") as f:
        import json
        json.dump(summary, f, indent=2)

    print("\nSummary saved to erccfs_multilabel_summary.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ERCCFS Multi-label Feature Analysis (32 features, 108 labels)")
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--feature-prefix", default="f", help="Prefix for feature columns")
    ap.add_argument("--label-prefix", default="y", help="Prefix for label columns")
    args = ap.parse_args()
    main(args.csv, args.feature_prefix, args.label_prefix)
