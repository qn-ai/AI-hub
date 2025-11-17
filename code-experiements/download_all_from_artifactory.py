# ==== sklearn_multitask_baselines.py ====
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
VAL_SIZE = 0.20
OUTPUT_DIR = Path("model_outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 0) DATA: replace with your actual frame & columns
# ---------------------------------------------------------------------
# df = pd.read_csv("your_data.csv")
# target_cols = [...]  # list of 108 target columns
# feature_cols = [c for c in df.columns if c not in target_cols]
participant_id_col = None  # e.g., "participant_id" if you have it

X = df[feature_cols].copy()
Y = df[target_cols].copy()

# ---------------------------------------------------------------------
# 1) TRAIN/VAL SPLIT (80/20)
# ---------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
)

# ---------------------------------------------------------------------
# 2) PREPROCESSING: numeric + categorical
# ---------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

pre = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

# Fit/transform
X_train_p = pre.fit_transform(X_train)
X_val_p   = pre.transform(X_val)

# ---------------------------------------------------------------------
# 3) METRICS
# ---------------------------------------------------------------------
from sklearn.metrics import f1_score

def macro_f1_per_target(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    scores = {col: f1_score(y_true_df[col], y_pred_df[col], average="macro")
              for col in y_true_df.columns}
    return scores, float(np.mean(list(scores.values())))

# ---------------------------------------------------------------------
# 4) (A) MULTIOUTPUTCLASSIFIER with HistGradientBoostingClassifier
#     (pure sklearn, native multiclass + predict_proba)
# ---------------------------------------------------------------------
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

base_hgb = HistGradientBoostingClassifier(
    max_depth=None, learning_rate=0.06, max_bins=255,
    l2_regularization=0.0, early_stopping=True,
    random_state=RANDOM_STATE
)

multi_hgb = MultiOutputClassifier(base_hgb, n_jobs=-1)
multi_hgb.fit(X_train_p, Y_train)

# Predict probabilities → labels + confidences
proba_list = multi_hgb.predict_proba(X_val_p)  # list of length = n_targets

hgb_preds = pd.DataFrame(index=Y_val.index)
hgb_out   = pd.DataFrame(index=Y_val.index)

for i, col in enumerate(target_cols):
    probs  = proba_list[i]             # (n_samples, n_classes_col)
    labels = probs.argmax(axis=1)
    confs  = probs.max(axis=1)
    hgb_preds[col] = labels
    hgb_out[f"{col}_label"] = labels
    hgb_out[f"{col}_conf"]  = confs

# Metrics
hgb_scores, hgb_mean_f1 = macro_f1_per_target(Y_val, hgb_preds)
print(f"[MultiOutput HGB] Mean Macro-F1 across {len(target_cols)} targets: {hgb_mean_f1:.4f}")

# Optional participant id
if participant_id_col and participant_id_col in df.columns:
    hgb_out.insert(0, participant_id_col, df.loc[Y_val.index, participant_id_col].values)

# Save CSV
hgb_csv = OUTPUT_DIR / "val_predictions_multioutput_hgb.csv"
hgb_out.to_csv(hgb_csv, index_label="row_index")
print(f"[MultiOutput HGB] Saved: {hgb_csv}")

# ---------------------------------------------------------------------
# 5) (B) CLASSIFIER CHAINS ENSEMBLE (captures label dependencies)
# ---------------------------------------------------------------------
from sklearn.multioutput import ClassifierChain

# We’ll reuse the same base model
n_chains = 5
chains = []
orders = []
for seed in range(n_chains):
    # random order lets each chain condition on different target permutations
    chain = ClassifierChain(
        base_estimator=HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.06, max_bins=255,
            l2_regularization=0.0, early_stopping=True,
            random_state=RANDOM_STATE + seed
        ),
        order="random",
        random_state=RANDOM_STATE + seed
    )
    chain.fit(X_train_p, Y_train)
    chains.append(chain)

# Predict proba from each chain, then average per target
# Each chain's predict_proba returns a list of arrays (one per target)
# We’ll collect per-target probas across chains, average, then argmax.
per_target_probas = [ [] for _ in target_cols ]  # list of lists

for chain in chains:
    chain_probas = chain.predict_proba(X_val_p)  # list length = n_targets
    for t, probs in enumerate(chain_probas):
        per_target_probas[t].append(probs)

# Average probas
cc_preds = pd.DataFrame(index=Y_val.index)
cc_out   = pd.DataFrame(index=Y_val.index)
for t, col in enumerate(target_cols):
    # Stack over chains: (n_chains, n_samples, n_classes_t)
    stacked = np.stack(per_target_probas[t], axis=0)
    avg_probs = stacked.mean(axis=0)  # (n_samples, n_classes_t)
    labels = avg_probs.argmax(axis=1)
    confs  = avg_probs.max(axis=1)
    cc_preds[col] = labels
    cc_out[f"{col}_label"] = labels
    cc_out[f"{col}_conf"]  = confs

# Metrics
cc_scores, cc_mean_f1 = macro_f1_per_target(Y_val, cc_preds)
print(f"[ClassifierChains (ensemble={n_chains})] Mean Macro-F1: {cc_mean_f1:.4f}")

if participant_id_col and participant_id_col in df.columns:
    cc_out.insert(0, participant_id_col, df.loc[Y_val.index, participant_id_col].values)

cc_csv = OUTPUT_DIR / "val_predictions_classifierchains_hgb.csv"
cc_out.to_csv(cc_csv, index_label="row_index")
print(f"[ClassifierChains] Saved: {cc_csv}")

# ---------------------------------------------------------------------
# 6) SUMMARY CSV
# ---------------------------------------------------------------------
summary = pd.DataFrame({
    "target": target_cols,
    "macro_f1_multioutput_hgb": [hgb_scores[c] for c in target_cols],
    "macro_f1_classifierchains": [cc_scores[c] for c in target_cols],
})
summary.loc["MEAN"] = ["<MEAN>", summary["macro_f1_multioutput_hgb"].mean(),
                       summary["macro_f1_classifierchains"].mean()]
summary_csv = OUTPUT_DIR / "validation_macro_f1_summary_sklearn.csv"
summary.to_csv(summary_csv, index=False)
print(f"[Summary] Saved: {summary_csv}")
