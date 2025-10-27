# evaluate_erccfs_numeric_stats.py
# Evaluate ERCCFS with accuracy/F1 plots, selection frequency, and significance tests.

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel, wilcoxon

from erccfs import make_placeholder_erccfs


# ---------- Utility ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))

def load_numeric_data(csv_path: str, feature_prefix: str, label_prefix: str, label_col: str = None):
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[feature_cols].to_numpy(dtype=float)
    feature_names = feature_cols

    if label_prefix:
        label_cols = [c for c in df.columns if c.startswith(label_prefix)]
        Y = df[label_cols].to_numpy(dtype=float)
        y_idx = Y.argmax(axis=1)
    elif label_col:
        y = df[label_col].to_numpy(dtype=int)
        classes = np.unique(y)
        Y = np.eye(len(classes))[np.searchsorted(classes, y)]
        y_idx = y
    else:
        raise ValueError("Provide either --label-prefix or --label-col.")

    return X, Y, y_idx, feature_names


# ---------- Evaluation ----------
def evaluate_erccfs(X, Y, y_idx, feature_names, k, folds, outdir, max_outer, max_inner, tol_inner):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    per_fold = []
    selected_sets = []
    selection_counts = np.zeros(X.shape[1], dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, y_idx), start=1):
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]

        model = make_placeholder_erccfs(
            n_features=k,
            stability_overlap=0.85,
            max_outer_iters=max_outer,
            max_inner_iters=max_inner,
            tol_inner=tol_inner,
        )
        model.fit(Xtr, Ytr)
        selected = model.selected_features_
        selected_sets.append(set(selected))
        selection_counts[selected] += 1

        clf = LogisticRegression(max_iter=2000, multi_class="ovr")

        # Selected features
        clf.fit(Xtr[:, selected], Ytr.argmax(axis=1))
        preds_sel = clf.predict(Xte[:, selected])
        acc_sel = accuracy_score(Yte.argmax(axis=1), preds_sel)
        f1_sel = f1_score(Yte.argmax(axis=1), preds_sel, average="macro")

        # All features
        clf.fit(Xtr, Ytr.argmax(axis=1))
        preds_all = clf.predict(Xte)
        acc_all = accuracy_score(Yte.argmax(axis=1), preds_all)
        f1_all = f1_score(Yte.argmax(axis=1), preds_all, average="macro")

        per_fold.append({
            "fold": fold,
            "n_selected": len(selected),
            "acc_selected": acc_sel,
            "f1_selected": f1_sel,
            "acc_all": acc_all,
            "f1_all": f1_all,
        })

    per_fold_df = pd.DataFrame(per_fold)

    # ---------- Statistical Tests ----------
    acc_sel = per_fold_df["acc_selected"]
    acc_all = per_fold_df["acc_all"]
    f1_sel = per_fold_df["f1_selected"]
    f1_all = per_fold_df["f1_all"]

    ttest_acc = ttest_rel(acc_all, acc_sel)
    ttest_f1 = ttest_rel(f1_all, f1_sel)
    wilcox_acc = wilcoxon(acc_all, acc_sel)
    wilcox_f1 = wilcoxon(f1_all, f1_sel)

    # ---------- Stability ----------
    jaccs = []
    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            jaccs.append(jaccard(selected_sets[i], selected_sets[j]))
    mean_jacc = float(np.mean(jaccs)) if jaccs else 1.0

    # ---------- Save metrics ----------
    sel_freq_df = pd.DataFrame({
        "feature": feature_names,
        "selection_count": selection_counts,
        "selection_rate": selection_counts / folds
    }).sort_values("selection_count", ascending=False)

    out_csv_metrics = os.path.join(outdir, "per_fold_metrics.csv")
    out_csv_freq = os.path.join(outdir, "selection_frequency.csv")
    per_fold_df.to_csv(out_csv_metrics, index=False)
    sel_freq_df.to_csv(out_csv_freq, index=False)

    # ---------- Plots ----------
    plt.figure(figsize=(7, 5))
    plt.plot(per_fold_df["fold"], per_fold_df["acc_selected"], marker='o', label="Selected Features")
    plt.plot(per_fold_df["fold"], per_fold_df["acc_all"], marker='x', label="All Features")
    plt.title("Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(per_fold_df["fold"], per_fold_df["f1_selected"], marker='o', label="Selected Features")
    plt.plot(per_fold_df["fold"], per_fold_df["f1_all"], marker='x', label="All Features")
    plt.title("Macro-F1 per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "f1_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    top_features = sel_freq_df.head(20)
    plt.bar(top_features["feature"], top_features["selection_rate"])
    plt.title("Top 20 Most Frequently Selected Features")
    plt.xlabel("Feature")
    plt.ylabel("Selection Rate")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "selection_frequency.png"), dpi=200)
    plt.close()

    # ---------- Summary ----------
    summary = {
        "k": int(k),
        "folds": int(folds),
        "mean_acc_selected": float(np.mean(acc_sel)),
        "mean_f1_selected": float(np.mean(f1_sel)),
        "mean_acc_all": float(np.mean(acc_all)),
        "mean_f1_all": float(np.mean(f1_all)),
        "mean_jaccard_stability": mean_jacc,
        "t_test": {
            "accuracy_p": float(ttest_acc.pvalue),
            "f1_p": float(ttest_f1.pvalue)
        },
        "wilcoxon": {
            "accuracy_p": float(wilcox_acc.pvalue),
            "f1_p": float(wilcox_f1.pvalue)
        },
        "metrics_csv": out_csv_metrics,
        "selection_frequency_csv": out_csv_freq,
        "plots": {
            "accuracy": "accuracy_comparison.png",
            "f1": "f1_comparison.png",
            "frequency": "selection_frequency.png"
        }
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate ERCCFS with plots and statistical tests")
    ap.add_argument("--csv", required=True, help="Path to numeric CSV")
    ap.add_argument("--feature-prefix", required=True, help="Prefix for feature columns (e.g., f)")
    ap.add_argument("--label-prefix", help="Prefix for one-hot label columns (e.g., y)")
    ap.add_argument("--label-col", help="Single class label column")
    ap.add_argument("--k", type=int, default=15, help="Top-k features to select")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--outdir", default="erccfs_eval_out", help="Output directory")
    ap.add_argument("--max-outer", type=int, default=10)
    ap.add_argument("--max-inner", type=int, default=100)
    ap.add_argument("--tol-inner", type=float, default=1e-5)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    X, Y, y_idx, feature_names = load_numeric_data(
        csv_path=args.csv,
        feature_prefix=args.feature_prefix,
        label_prefix=args.label_prefix,
        label_col=args.label_col,
    )

    summary = evaluate_erccfs(
        X=X,
        Y=Y,
        y_idx=y_idx,
        feature_names=feature_names,
        k=args.k,
        folds=args.folds,
        outdir=outdir,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        tol_inner=args.tol_inner,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
