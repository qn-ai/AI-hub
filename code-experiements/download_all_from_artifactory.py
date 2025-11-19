import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.cluster import KMeans

# ============================================================
# CONFIG
# ============================================================
PRED_FILE = "input_with_predictions.csv"
CV_FILE = "per_target_results_cv.csv"
OUTPUT_METRICS = "per_target_full_metrics.csv"

# A target is imbalanced if the majority class exceeds this threshold
IMBALANCE_THRESHOLD = 0.80


def compute_metrics():
    print("Loading predictions…")
    df = pd.read_csv(PRED_FILE)
    cv = pd.read_csv(CV_FILE)

    target_cols = [
        col for col in df.columns
        if col.startswith("y_") and not col.endswith("_pred")
    ]

    metrics = []

    for target in target_cols:
        pred_col = target + "_pred"
        if pred_col not in df.columns:
            continue

        # rows with true labels
        mask = df[target].notna()
        y_true = df.loc[mask, target]
        y_pred = df.loc[mask, pred_col]

        # detect imbalanced target
        value_counts = y_true.value_counts(normalize=True)
        max_class_prop = value_counts.max()
        is_imbalanced = max_class_prop > IMBALANCE_THRESHOLD

        # Compute metrics (macro handles multi-class)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics.append({
            "target": target,
            "n_labelled": len(y_true),
            "is_imbalanced": is_imbalanced,
            "majority_class_prop": max_class_prop,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.merge(cv, on="target", how="left")
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    print(f"Saved full metrics → {OUTPUT_METRICS}")
    return metrics_df


def plot_accuracy_vs_f1(metrics_df):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        metrics_df["accuracy"],
        metrics_df["f1_macro"],
        c=metrics_df["is_imbalanced"].map({True: "red", False: "blue"}),
        alpha=0.7
    )
    plt.xlabel("Accuracy")
    plt.ylabel("F1-score (macro)")
    plt.title("Accuracy vs F1-score per Target")
    plt.grid(True)
    plt.savefig("plot_accuracy_vs_f1.png", dpi=200)
    print("Saved plot → plot_accuracy_vs_f1.png")


def cluster_targets(metrics_df, n_clusters=3):
    # Use accuracy and F1 for clustering
    features = metrics_df[["accuracy", "f1_macro"]].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    metrics_df["cluster"] = labels
    metrics_df.to_csv("per_target_full_metrics_with_clusters.csv", index=False)
    print("Saved clusters → per_target_full_metrics_with_clusters.csv")

    # Plot clusters
    plt.figure(figsize=(10, 6))
    for cluster_id in range(n_clusters):
        subset = metrics_df[metrics_df["cluster"] == cluster_id]
        plt.scatter(
            subset["accuracy"],
            subset["f1_macro"],
            label=f"Cluster {cluster_id}",
            alpha=0.7
        )

    plt.xlabel("Accuracy")
    plt.ylabel("F1-score")
    plt.title("Target Clusters by Accuracy & F1-score")
    plt.legend()
    plt.grid(True)
    plt.savefig("cluster_accuracy_f1.png", dpi=200)
    print("Saved cluster plot → cluster_accuracy_f1.png")


def recommend_targets(metrics_df):
    good = metrics_df[
        (metrics_df["f1_macro"] > 0.80) &
        (~metrics_df["is_imbalanced"])
    ]

    medium = metrics_df[
        (metrics_df["f1_macro"].between(0.60, 0.80))
    ]

    bad = metrics_df[
        (metrics_df["f1_macro"] < 0.60)
    ]

    good.to_csv("recommended_good_targets.csv", index=False)
    medium.to_csv("recommended_medium_targets.csv", index=False)
    bad.to_csv("recommended_bad_targets.csv", index=False)

    print("Saved recommendations:")
    print(" → recommended_good_targets.csv")
    print(" → recommended_medium_targets.csv")
    print(" → recommended_bad_targets.csv")


if __name__ == "__main__":
    metrics = compute_metrics()
    plot_accuracy_vs_f1(metrics)
    cluster_targets(metrics)
    recommend_targets(metrics)
