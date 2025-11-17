# ---------------------------------------------------------------------
# 4) (A) MULTIOUTPUTCLASSIFIER with HistGradientBoostingClassifier
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
    early_stopping=False,  # IMPORTANT: prevents class=1 errors
    random_state=RANDOM_STATE
)

multi_hgb = MultiOutputClassifier(base_hgb, n_jobs=-1)
multi_hgb.fit(X_train_p, Y_train)

# Predict probabilities → labels + confidences
proba_list = multi_hgb.predict_proba(X_val_p)

hgb_preds = pd.DataFrame(index=Y_val.index)
hgb_out   = pd.DataFrame(index=Y_val.index)

for i, col in enumerate(target_cols):
    probs   = proba_list[i]
    est     = multi_hgb.estimators_[i]
    classes = est.classes_

    max_idx = probs.argmax(axis=1)
    labels  = classes[max_idx]
    confs   = probs[np.arange(len(max_idx)), max_idx]

    hgb_preds[col] = labels
    hgb_out[f"{col}_label"] = labels
    hgb_out[f"{col}_conf"]  = confs


# -------------------------------
# 5) MERGE back to original input
# -------------------------------

# Extract id_pwd_id (must exist in input_data)
val_ids = input_data.loc[Y_val.index, "id_pwd_id"]

# Insert id at the front
hgb_out.insert(0, "id_pwd_id", val_ids)

# Merge side-by-side with input_data rows
merged_val_results = pd.concat(
    [
        input_data.loc[Y_val.index].reset_index(drop=True),
        hgb_out.reset_index(drop=True)
    ],
    axis=1
)

# -------------------------------
# Save final merged CSV
# -------------------------------
final_csv = OUTPUT_DIR / "final_validation_results_with_id_pwd_id.csv"
merged_val_results.to_csv(final_csv, index=False)

print(f"[Final Output] Saved merged results with id_pwd_id → {final_csv}")
