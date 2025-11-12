# === multi_output_confidence_export.py ===
import os
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# 0) CONFIG / INPUT
# -----------------------
RANDOM_STATE = 42
VAL_SIZE = 0.20
N_ESTIMATORS = 400  # LightGBM
OUTPUT_DIR = Path("model_outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

# Load your data ---------------------------------------------------------------
# TODO: replace with your actual data source:
# df = pd.read_csv("your_data.csv")
# For illustration, assume df is already in memory.
# df = ...

# Identify columns -------------------------------------------------------------
# TODO: provide your list of 108 target columns:
# target_cols = [...]
# If you have a naming pattern, e.g. startswith("t_"), you could do:
# target_cols = [c for c in df.columns if c.startswith("target_")]

# Features = everything else
# feature_cols = [c for c in df.columns if c not in target_cols]

# Optional: participant id if you have one (used only for output)
# TODO: set to None or a column name like "participant_id"
participant_id_col = None  # e.g., "participant_id"

# -----------------------
# 1) TRAIN/VAL SPLIT
# -----------------------
from sklearn.model_selection import train_test_split

X = df[feature_cols].copy()
Y = df[target_cols].copy()

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
)

# -----------------------
# 2) PREPROCESSING
# -----------------------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Detect feature types
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

# Fit on train only; transform train and val
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep   = preprocessor.transform(X_val)

# -----------------------
# 3) METRICS (Macro-F1)
# -----------------------
from sklearn.metrics import f1_score

def macro_f1_per_target(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    scores = {}
    for col in y_true_df.columns:
        scores[col] = f1_score(y_true_df[col], y_pred_df[col], average="macro")
    return scores, np.mean(list(scores.values()))

# -----------------------
# 4) MODEL A: LIGHTGBM (MultiOutput)
# -----------------------
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

lgb_base = LGBMClassifier(
    objective="multiclass",
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lgb_model = MultiOutputClassifier(lgb_base, n_jobs=-1)
lgb_model.fit(X_train_prep, Y_train)

# Predict probabilities → labels + confidences
# predict_proba returns a list of length = n_targets
proba_list = lgb_model.predict_proba(X_val_prep)

# Build validation outputs (LightGBM)
lgb_pred = pd.DataFrame(index=Y_val.index)
lgb_out  = pd.DataFrame(index=Y_val.index)

for i, col in enumerate(target_cols):
    probs = proba_list[i]  # shape (n_samples, n_classes_target_i)
    labels = probs.argmax(axis=1)
    confs  = probs.max(axis=1)

    lgb_pred[col] = labels
    lgb_out[f"{col}_label"] = labels
    lgb_out[f"{col}_conf"]  = confs

# Add participant id if available
if participant_id_col is not None and participant_id_col in df.columns:
    lgb_out.insert(0, participant_id_col, df.loc[Y_val.index, participant_id_col].values)

# Evaluate
lgb_scores, lgb_mean_f1 = macro_f1_per_target(Y_val, lgb_pred)

print(f"[LightGBM] Mean Macro-F1 across {len(target_cols)} targets: {lgb_mean_f1:.4f}")

# Save CSV
lgb_csv_path = OUTPUT_DIR / "val_predictions_lightgbm.csv"
lgb_out.to_csv(lgb_csv_path, index_label="row_index")
print(f"[LightGBM] Validation predictions saved to: {lgb_csv_path}")

# -----------------------
# 5) MODEL B: PYTORCH MULTI-TASK NN
# -----------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine output dims per target based on train set
target_nclasses = [Y_train[c].nunique() for c in target_cols]

class ArrayDataset(Dataset):
    def __init__(self, X_csr, Y_df):
        self.X = X_csr  # can be csr or ndarray
        self.y = Y_df.values.astype(np.int64)

        # If sparse: convert to dense only once, keeping memory in mind
        if hasattr(self.X, "toarray"):
            self.X = self.X.toarray().astype(np.float32)
        else:
            self.X = self.X.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

input_dim = X_train_prep.shape[1]

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.35, out_dims=None):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim//2, k) for k in out_dims])

    def forward(self, x):
        z = self.shared(x)
        logits = [head(z) for head in self.heads]
        return logits

net = MultiTaskNet(input_dim, hidden_dim=512, dropout=0.35, out_dims=target_nclasses).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)

train_ds = ArrayDataset(X_train_prep, Y_train)
val_ds   = ArrayDataset(X_val_prep, Y_val)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

def eval_on_loader(model, loader):
    model.eval()
    all_preds = [[] for _ in target_cols]
    all_confs = [[] for _ in target_cols]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits_list = model(xb)
            for t, logits in enumerate(logits_list):
                probs = torch.softmax(logits, dim=1)  # (B, C_t)
                conf, pred = probs.max(dim=1)
                all_preds[t].append(pred.cpu().numpy())
                all_confs[t].append(conf.cpu().numpy())
    # stack
    preds_df = pd.DataFrame(index=Y_val.index)
    out_df   = pd.DataFrame(index=Y_val.index)
    for t, col in enumerate(target_cols):
        preds = np.concatenate(all_preds[t])
        confs = np.concatenate(all_confs[t])
        preds_df[col] = preds
        out_df[f"{col}_label"] = preds
        out_df[f"{col}_conf"]  = confs
    return preds_df, out_df

# Training with early stopping
best_val_f1 = -np.inf
patience = 12
bad_epochs = 0
EPOCHS = 120

for epoch in range(1, EPOCHS+1):
    net.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits_list = net(xb)

        # sum of cross-entropies across 108 targets
        loss = 0.0
        for t in range(len(target_cols)):
            loss = loss + criterion(logits_list[t], yb[:, t])

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluate
    nn_preds_df, nn_out_df = eval_on_loader(net, val_loader)
    nn_scores, nn_mean_f1 = macro_f1_per_target(Y_val, nn_preds_df)

    print(f"[NN] Epoch {epoch:03d} | loss={running_loss:.2f} | val_mean_macroF1={nn_mean_f1:.4f}")

    # Early stopping
    if nn_mean_f1 > best_val_f1 + 1e-4:
        best_val_f1 = nn_mean_f1
        bad_epochs = 0
        best_state = {k: v.cpu() for k, v in net.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"[NN] Early stopping at epoch {epoch}. Best mean Macro-F1: {best_val_f1:.4f}")
            break

# Restore best model
if 'best_state' in locals():
    net.load_state_dict(best_state)

# Final validation outputs for NN
nn_preds_df, nn_out_df = eval_on_loader(net, val_loader)
nn_scores, nn_mean_f1 = macro_f1_per_target(Y_val, nn_preds_df)
print(f"[NN] Final Mean Macro-F1 across {len(target_cols)} targets: {nn_mean_f1:.4f}")

# Add participant id if available
if participant_id_col is not None and participant_id_col in df.columns:
    nn_out_df.insert(0, participant_id_col, df.loc[Y_val.index, participant_id_col].values)

# Save CSV
nn_csv_path = OUTPUT_DIR / "val_predictions_multitask_nn.csv"
nn_out_df.to_csv(nn_csv_path, index_label="row_index")
print(f"[NN] Validation predictions saved to: {nn_csv_path}")

# -----------------------
# 6) OPTIONAL: SUMMARY CSV OF METRICS
# -----------------------
summary = pd.DataFrame({
    "target": target_cols,
    "macro_f1_lightgbm": [lgb_scores[c] for c in target_cols],
    "macro_f1_nn":       [nn_scores[c]  for c in target_cols],
})
summary.loc["MEAN"] = ["<MEAN>", summary["macro_f1_lightgbm"].mean(), summary["macro_f1_nn"].mean()]
summary_path = OUTPUT_DIR / "validation_macro_f1_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"[Summary] Macro-F1 by target saved to: {summary_path}")
