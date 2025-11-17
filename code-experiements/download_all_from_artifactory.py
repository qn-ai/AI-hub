# Metrics
hgb_scores, hgb_mean_f1 = macro_f1_per_target(Y_val, hgb_preds)

# Accuracy (per target)
acc_dict, mean_acc = accuracy_per_target(Y_val, hgb_preds)

# Accuracy (per row exact match)
exact_acc = exact_match_accuracy(Y_val, hgb_preds)

print(f"[MultiOutput HGB] Mean Macro-F1: {hgb_mean_f1:.4f}")
print(f"[MultiOutput HGB] Mean per-target accuracy: {mean_acc:.4f}")
print(f"[MultiOutput HGB] Exact-match accuracy: {exact_acc:.4f}")
