#!/usr/bin/env python
"""
Full multi-target evaluation + HTML report.

Inputs (already produced by your training pipeline):
- input_with_predictions.csv
- per_target_results_cv.csv

Outputs:
- per_target_full_metrics.csv
- per_target_full_metrics_with_clusters.csv
- recommended_good_targets.csv
- recommended_medium_targets.csv
- recommended_bad_targets.csv
- plot_accuracy_vs_f1.png
- cluster_accuracy_f1.png
- report_targets.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.cluster import KMeans

# ============================================================
# CONFIG
# ============================================================
PRED_FILE = "input_with_predictions.csv"
CV_FILE = "per_target_results_cv.csv"

OUTPUT_METRICS = "per_target_full_metrics.csv"
OUTPUT_METRICS_CLUST = "per_target_full_metrics_with_clusters.csv"
REPORT_HTML = "report_targets.html"

IMBALANCE_THRESHOLD = 0.80  # majority class share to flag imbalance
N_CLUSTERS = 3              # clusters for accuracy+F1


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_metrics(pred_file=PRED_FILE, cv_file=CV_FILE) -> pd.DataFrame:
    print(f"Loading predictions from: {pred_file}")
    df = pd.read_csv(pred_file)

    print(f"Loading CV summary from: {cv_file}")
    cv = pd.read_csv(cv_file)

    # All true target columns (y_*, excluding *_pred)
    target_cols = [
        col for col in df.columns
        if col.startswith("y_") and not col.endswith("_pred")
    ]
    print(f"Found {len(target_cols)} target columns with true labels.")

    rows = []

    for target in target_cols:
        pred_col = target + "_pred"
        if pred_col not in df.columns:
            print(f"  [WARN] Missing prediction column for {target}, skipping.")
            continue

        mask = df[target].notna()
        y_true = df.loc[mask, target]
        y_pred = df.loc[mask, pred_col]

        if len(y_true) == 0:
            print(f"  [WARN] No labelled rows for {target}, skipping.")
            continue

        # Imbalance detection
        value_counts = y_true.value_counts(normalize=True)
        max_prop = value_counts.max()
        is_imbalanced = bool(max_prop > IMBALANCE_THRESHOLD)

        # Macro metrics (works for multi-class)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        rec = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        rows.append(
            {
                "target": target,
                "n_labelled_eval": len(y_true),
                "is_imbalanced": is_imbalanced,
                "majority_class_prop": max_prop,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
            }
        )

    metrics_df = pd.DataFrame(rows)
    print(f"Computed metrics for {len(metrics_df)} targets.")

    # Merge with CV info
    metrics_df = metrics_df.merge(cv, on="target", how="left")

    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved full per-target metrics → {OUTPUT_METRICS}")

    return metrics_df


# ============================================================
# PLOTS & CLUSTERS
# ============================================================

def plot_accuracy_vs_f1(metrics_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    colors = metrics_df["is_imbalanced"].map({True: "red", False: "blue"})
    plt.scatter(
        metrics_df["accuracy"],
        metrics_df["f1_macro"],
        c=colors,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )

    plt.xlabel("Accuracy (eval on labelled rows)")
    plt.ylabel("F1-score (macro)")
    plt.title("Accuracy vs F1-score per target")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_accuracy_vs_f1.png", dpi=200)
    plt.close()
    print("Saved plot → plot_accuracy_vs_f1.png")


def cluster_targets(metrics_df: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    features = metrics_df[["accuracy", "f1_macro"]].fillna(0.0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(features)

    metrics_df = metrics_df.copy()
    metrics_df["cluster"] = labels
    metrics_df.to_csv(OUTPUT_METRICS_CLUST, index=False)
    print(f"Saved metrics with clusters → {OUTPUT_METRICS_CLUST}")

    # Plot clusters
    plt.figure(figsize=(10, 6))
    for cluster_id in range(n_clusters):
        subset = metrics_df[metrics_df["cluster"] == cluster_id]
        plt.scatter(
            subset["accuracy"],
            subset["f1_macro"],
            alpha=0.7,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Accuracy")
    plt.ylabel("F1-score (macro)")
    plt.title("Target clusters by Accuracy & F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cluster_accuracy_f1.png", dpi=200)
    plt.close()
    print("Saved cluster plot → cluster_accuracy_f1.png")

    return metrics_df


# ============================================================
# RECOMMENDATIONS
# ============================================================

def recommend_targets(metrics_df: pd.DataFrame):
    good = metrics_df[
        (metrics_df["f1_macro"] >= 0.80)
        & (~metrics_df["is_imbalanced"])
    ]
    medium = metrics_df[
        (metrics_df["f1_macro"] < 0.80)
        & (metrics_df["f1_macro"] >= 0.60)
    ]
    bad = metrics_df[metrics_df["f1_macro"] < 0.60]

    good.to_csv("recommended_good_targets.csv", index=False)
    medium.to_csv("recommended_medium_targets.csv", index=False)
    bad.to_csv("recommended_bad_targets.csv", index=False)

    print("Saved recommendations:")
    print("  → recommended_good_targets.csv")
    print("  → recommended_medium_targets.csv")
    print("  → recommended_bad_targets.csv")

    return good, medium, bad


# ============================================================
# HTML REPORT
# ============================================================

def df_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    df_small = df.head(max_rows).copy()
    return df_small.to_html(
        index=False,
        border=0,
        justify="left",
        float_format=lambda x: f"{x:0.4f}" if isinstance(x, float) else x,
        classes="table table-sm table-striped",
    )


def build_html_report(
    metrics_df: pd.DataFrame,
    good: pd.DataFrame,
    medium: pd.DataFrame,
    bad: pd.DataFrame,
    report_path: str = REPORT_HTML,
):
    n_targets = len(metrics_df)
    n_good = len(good)
    n_medium = len(medium)
    n_bad = len(bad)
    n_imbalanced = metrics_df["is_imbalanced"].sum()

    avg_acc = metrics_df["accuracy"].mean()
    avg_f1 = metrics_df["f1_macro"].mean()

    # Sort for display
    best_by_f1 = metrics_df.sort_values("f1_macro", ascending=False)
    worst_by_f1 = metrics_df.sort_values("f1_macro", ascending=True)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Multi-target classification report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
}}
h1, h2, h3 {{
    color: #333;
}}
.table {{
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
}}
.table th, .table td {{
    border: 1px solid #ddd;
    padding: 4px 6px;
    font-size: 12px;
}}
.table th {{
    background-color: #f2f2f2;
}}
.bad {{ color: #b30000; }}
.good {{ color: #006600; }}
.note {{ font-size: 12px; color: #666; }}
</style>
</head>
<body>

<h1>Multi-target classification report</h1>

<h2>Overview</h2>
<ul>
  <li>Total targets evaluated: <b>{n_targets}</b></li>
  <li>Average accuracy: <b>{avg_acc:0.4f}</b></li>
  <li>Average F1 (macro): <b>{avg_f1:0.4f}</b></li>
  <li>Imbalanced targets (majority class &gt; {IMBALANCE_THRESHOLD:.0%}): 
      <b>{n_imbalanced}</b></li>
  <li>Recommended GOOD targets (F1 ≥ 0.80 &amp; not imbalanced): 
      <b>{n_good}</b></li>
  <li>Recommended MEDIUM targets (0.60 ≤ F1 &lt; 0.80): 
      <b>{n_medium}</b></li>
  <li>Recommended BAD targets (F1 &lt; 0.60): 
      <b>{n_bad}</b></li>
</ul>

<h2>Global plots</h2>
<p>
  <b>Accuracy vs F1-score:</b><br>
  <img src="plot_accuracy_vs_f1.png" alt="Accuracy vs F1" style="max-width: 100%; border:1px solid #ccc;">
</p>
<p>
  <b>Target clusters by Accuracy &amp; F1:</b><br>
  <img src="cluster_accuracy_f1.png" alt="Clusters Accuracy vs F1" style="max-width: 100%; border:1px solid #ccc;">
</p>

<h2>Best targets by F1-score</h2>
<p class="note">Top targets ranked by F1-score (macro). These are generally the most reliable.</p>
{df_to_html_table(best_by_f1[[
    "target", "n_labelled_eval", "accuracy", "f1_macro",
    "precision_macro", "recall_macro",
    "is_imbalanced", "majority_class_prop"
]])}

<h2>Worst targets by F1-score</h2>
<p class="note">Targets with the lowest F1-score. Use with caution or review label quality.</p>
{df_to_html_table(worst_by_f1[[
    "target", "n_labelled_eval", "accuracy", "f1_macro",
    "precision_macro", "recall_macro",
    "is_imbalanced", "majority_class_prop"
]])}

<h2>Recommended GOOD targets</h2>
<p class="note">F1 ≥ 0.80 and not flagged as imbalanced. Suitable for downstream use and imputation.</p>
{df_to_html_table(good[[
    "target", "n_labelled_eval", "accuracy", "f1_macro",
    "precision_macro", "recall_macro",
    "cv_mean_acc", "cv_std_acc",
    "is_imbalanced", "majority_class_prop"
]])}

<h2>Recommended MEDIUM targets</h2>
<p class="note">F1 between 0.60 and 0.80. Usable but predictions may be less reliable.</p>
{df_to_html_table(medium[[
    "target", "n_labelled_eval", "accuracy", "f1_macro",
    "precision_macro", "recall_macro",
    "cv_mean_acc", "cv_std_acc",
    "is_imbalanced", "majority_class_prop"
]])}

<h2>Recommended BAD targets</h2>
<p class="note bad">F1 &lt; 0.60. Predictions are weak or labels may be noisy; generally not recommended for production use.</p>
{df_to_html_table(bad[[
    "target", "n_labelled_eval", "accuracy", "f1_macro",
    "precision_macro", "recall_macro",
    "cv_mean_acc", "cv_std_acc",
    "is_imbalanced", "majority_class_prop"
]])}

<hr>
<p class="note">
Generated automatically by generate_full_report.py.
You can print this page to PDF from your browser if you need a PDF report.
</p>

</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved HTML report → {report_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    metrics_df = compute_metrics()
    plot_accuracy_vs_f1(metrics_df)
    metrics_with_clusters = cluster_targets(metrics_df)
    good, medium, bad = recommend_targets(metrics_with_clusters)
    build_html_report(metrics_with_clusters, good, medium, bad)


if __name__ == "__main__":
    main()
