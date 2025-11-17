# -------------------------------
# Accuracy per target + summary
# -------------------------------
# Per-target accuracy
hgb_acc_dict = accuracy_per_target(Y_val, hgb_preds)

# Convert to Series for easy stats
acc_series = pd.Series(hgb_acc_dict)

mean_acc   = acc_series.mean()
median_acc = acc_series.median()
std_acc    = acc_series.std()

min_acc    = acc_series.min()
max_acc    = acc_series.max()
min_targets = acc_series[acc_series == min_acc].index.tolist()
max_targets = acc_series[acc_series == max_acc].index.tolist()

print(f"[MultiOutput HGB] Mean per-target accuracy: {mean_acc:.4f}")
print(f"[MultiOutput HGB] Median per-target accuracy: {median_acc:.4f}")
print(f"[MultiOutput HGB] Std of per-target accuracy: {std_acc:.4f}")
print(f"[MultiOutput HGB] Min accuracy: {min_acc:.4f} → {min_targets}")
print(f"[MultiOutput HGB] Max accuracy: {max_acc:.4f} → {max_targets}")

# -------------------------------
# Build and save accuracy/F1 summary CSV
# -------------------------------
summary_df = pd.DataFrame({
    "target": target_cols,
    "macro_f1": [hgb_scores[c] for c in target_cols],
    "accuracy": [hgb_acc_dict[c] for c in target_cols],
})

# Add global stats as an extra row (optional)
summary_df.loc[len(summary_df)] = [
    "<SUMMARY>",
    summary_df["macro_f1"].mean(),
    summary_df["accuracy"].mean(),
]

summary_path = OUTPUT_DIR / "validation_metrics_summary_hgb.csv"
summary_df.to_csv(summary_path, index=False)
print(f"[MultiOutput HGB] Metrics summary saved to: {summary_path}")

# -------------------------------
# Rank targets by difficulty
# -------------------------------
rank_df = pd.DataFrame({
    "target": target_cols,
    "accuracy": [hgb_acc_dict[c] for c in target_cols],
    "macro_f1": [hgb_scores[c]   for c in target_cols],
})

# Sort by accuracy ascending (hardest first)
rank_df = rank_df.sort_values("accuracy", ascending=True).reset_index(drop=True)

# Show top 10 hardest targets in console
print("\n=== Hardest targets (lowest accuracy) ===")
print(rank_df.head(10))

# Show top 10 easiest targets
print("\n=== Easiest targets (highest accuracy) ===")
print(rank_df.tail(10))

# Save ranked list to CSV
rank_path = OUTPUT_DIR / "targets_ranked_by_accuracy_hgb.csv"
rank_df.to_csv(rank_path, index=False)
print(f"\n[MultiOutput HGB] Ranked targets saved to: {rank_path}")


# ---------------------------------------------
# Class distribution for the hardest targets
# ---------------------------------------------
num_hardest = 5   # change to 10 if you want more

hardest_targets = rank_df.head(num_hardest)["target"].tolist()

print("\n=== Class distributions for hardest targets ===")
classdist_list = []   # for CSV export

for col in hardest_targets:
    print(f"\n--- {col} ---")
    
    # counts from validation set
    value_counts = Y_val[col].value_counts(dropna=False)
    value_perc   = Y_val[col].value_counts(normalize=True, dropna=False)
    
    # print nicely
    print(pd.DataFrame({
        "count": value_counts,
        "percentage": value_perc
    }))
    
    # store for CSV
    temp_df = pd.DataFrame({
        "target": col,
        "class": value_counts.index,
        "count": value_counts.values,
        "percentage": value_perc.values,
    })
    classdist_list.append(temp_df)

# Combine & save CSV
classdist_df = pd.concat(classdist_list, ignore_index=True)

dist_path = OUTPUT_DIR / "class_distributions_hardest_targets.csv"
classdist_df.to_csv(dist_path, index=False)

print(f"\n[MultiOutput HGB] Class distribution summary saved to: {dist_path}")
