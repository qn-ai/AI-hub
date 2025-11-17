# ---------------------------------------------------------------------
# 4) (A) MULTIOUTPUTCLASSIFIER with HistGradientBoostingClassifier
#     (pure sklearn, native multiclass + predict_proba)
# ---------------------------------------------------------------------
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pandas as pd

base_hgb = HistGradientBoostingClassifier(
    max_depth=None,
    learning_rate=0.06,
    max_bins=255,
    l2_regularization=0.0,
    early_stopping=False,        # ✅ turn off internal stratified split
    random_state=RANDOM_STATE
)

multi_hgb = MultiOutputClassifier(base_hgb, n_jobs=-1)
multi_hgb.fit(X_train_p, Y_train)

# Predict probabilities → labels + confidences
proba_list = multi_hgb.predict_proba(X_val_p)  # list of length = n_targets

hgb_preds = pd.DataFrame(index=Y_val.index)
hgb_out   = pd.DataFrame(index=Y_val.index)

for i, col in enumerate(target_cols):
    probs = proba_list[i]              # shape (n_samples, n_classes_for_this_target)

    # ✅ get the underlying estimator & its original class labels
    est = multi_hgb.estimators_[i]
    classes = est.classes_             # e.g. array(['A','B','C',...]) or ints

    # indices of max probability
    max_idx = probs.argmax(axis=1)     # 0..n_classes-1

    # ✅ map indices → original labels (avoid mix of label types)
    labels = classes[max_idx]

    # confidence = probability of predicted class
    confs = probs[np.arange(probs.shape[0]), max_idx]

    # store
    hgb_preds[col] = labels
    hgb_out[f"{col}_label"] = labels
    hgb_out[f"{col}_conf"]  = confs

# Metrics (now y_true and y_pred have same label types)
hgb_scores, hgb_mean_f1 = macro_f1_per_target(Y_val, hgb_preds)
print(f"[MultiOutput HGB] Mean Macro-F1 across {len(target_cols)} targets: {hgb_mean_f1:.4f}")

# Optional participant id
if participant_id_col and participant_id_col in df.columns:
    hgb_out.insert(0, participant_id_col, df.loc[Y_val.index, participant_id_col].values)

# Save CSV
hgb_csv = OUTPUT_DIR / "val_predictions_multioutput_hgb.csv"
hgb_out.to_csv(hgb_csv, index_label="row_index")
print(f"[MultiOutput HGB] Saved: {hgb_csv}")
