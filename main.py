# dtype_utils.py
"""
Reusable pandas dtype utilities for ML pipelines.

Design goals:
- Never let object dtypes reach models
- Clean ordinal targets safely
- Coerce numeric-like strings
- Drop unusable features deterministically
- Fail fast if invariants are violated
"""

import numpy as np
import pandas as pd


# --------------------------------------------------
# Ordinal target cleaning (CORAL-safe)
# --------------------------------------------------
def clean_ordinal_targets(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    Extract leading integer from ordinal target labels.
    Example: '0.never' -> 0, '2.high' -> 2

    Rows with invalid targets become NaN and are dropped.
    """
    df = df.copy()

    def _clean(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
             .str.extract(r"^(\d+)", expand=False)
             .astype(float)
        )

    for t in targets:
        if t not in df.columns:
            raise KeyError(f"Target column not found: {t}")
        df[t] = _clean(df[t])

    # drop rows with invalid targets
    df = df.dropna(subset=targets)

    # enforce integer dtype
    df[targets] = df[targets].astype(int)

    return df


# --------------------------------------------------
# Feature dtype cleaning
# --------------------------------------------------
def fix_feature_dtypes(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    numeric_ratio_threshold: float = 0.80,
    min_non_null_ratio: float = 0.20,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Fix bad pandas dtypes in feature columns.

    Strategy:
    - object columns:
        - if >= numeric_ratio_threshold numeric-like -> coerce to numeric
        - else -> drop
    - drop columns mostly NaN after coercion
    - replace inf/-inf
    - fill NaN with 0.0

    Returns:
        df_clean
        new_feature_cols
        report (dict for logging/debugging)
    """
    df = df.copy()

    obj_cols = [c for c in feature_cols if c in df.columns and df[c].dtype == "object"]

    def numeric_like_ratio(s: pd.Series, sample_size: int = 2000) -> float:
        x = s.dropna().astype(str).head(sample_size)
        if len(x) == 0:
            return 0.0
        x = (
            x.str.replace(",", "", regex=False)
             .str.strip()
        )
        return float(
            x.str.match(r"^-?\d+(\.\d+)?$", na=False).mean()
        )

    numeric_like = []
    non_numeric = []

    for c in obj_cols:
        r = numeric_like_ratio(df[c])
        if r >= numeric_ratio_threshold:
            numeric_like.append(c)
        else:
            non_numeric.append(c)

    # --- coerce numeric-like object columns
    for c in numeric_like:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- drop non-numeric object columns
    drop_cols = list(non_numeric)

    # --- drop columns that are mostly NaN after coercion
    for c in feature_cols:
        if c in df.columns:
            if df[c].notna().mean() < min_non_null_ratio:
                drop_cols.append(c)

    drop_cols = sorted(set(drop_cols))
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    new_feature_cols = [c for c in feature_cols if c not in drop_cols]

    # --- final safety check
    still_object = [c for c in new_feature_cols if df[c].dtype == "object"]
    if still_object:
        raise ValueError(f"Object dtypes remain after cleaning: {still_object}")

    # --- LightGBM-safe cleanup
    df[new_feature_cols] = (
        df[new_feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    report = {
        "object_features_initial": obj_cols,
        "numeric_like_converted": numeric_like,
        "non_numeric_dropped": non_numeric,
        "dropped_low_signal": [c for c in drop_cols if c not in non_numeric],
        "final_feature_count": len(new_feature_cols),
    }

    return df, new_feature_cols, report


# --------------------------------------------------
# Assertions (fail fast)
# --------------------------------------------------
def assert_numeric_features(df: pd.DataFrame, feature_cols: list[str]):
    bad = [c for c in feature_cols if df[c].dtype == "object"]
    if bad:
        raise AssertionError(f"Non-numeric features detected: {bad}")


def assert_integer_targets(df: pd.DataFrame, targets: list[str]):
    for t in targets:
        if not pd.api.types.is_integer_dtype(df[t]):
            raise AssertionError(f"Target '{t}' is not integer dtype")



# coral_train.py
import os
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from dtype_utils import (
    clean_ordinal_targets,
    fix_feature_dtypes,
    assert_numeric_features,
    assert_integer_targets,
)

# =============================
# CONFIG (EDIT HERE)
# =============================
TRAIN_CSV = "training_data.csv"
TARGETS = ["y_target1", "y_target2", "y_target3"]   # <-- EDIT
ID_COL = "record_id"                                # optional, or None

ARTIFACT_DIR = "artifacts"
MODEL_DIR = f"{ARTIFACT_DIR}/coral_models"
STAGE0_PATH = f"{ARTIFACT_DIR}/stage0_features.joblib"
RUN_REPORT_PATH = f"{ARTIFACT_DIR}/dtype_report_train.json"

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Decode threshold tuning
DECODE_GRID = [0.4, 0.5, 0.6]
ALPHA = 0.7

# Improved weight config (single business-safe default)
WEIGHT_CFG = {"power": 0.5, "cap": 12.0, "gamma": 0.3}

# LightGBM base params (fast-ish defaults)
LGB_BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 42,
}

NUM_BOOST_ROUND = 1200
EARLY_STOPPING_ROUNDS = 80


# =============================
# CORAL helpers
# =============================
def make_coral_targets(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    thr = np.arange(K - 1)
    return (y[:, None] > thr[None, :]).astype(np.int8)

def decode_coral(proba_gt: np.ndarray, t: float) -> np.ndarray:
    return (proba_gt >= float(t)).sum(axis=1).astype(int)

def dist_pct(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    c = np.bincount(y, minlength=K).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.zeros(K, dtype=float)

def compute_spw(pos: float, neg: float, cap: float, power: float) -> float:
    if pos <= 0:
        return 1.0
    return float(min((neg / pos) ** power, cap))

def tail_weight(k: int, K: int, gamma: float) -> float:
    return float(1.0 + gamma * (k / (K - 1))) if K > 1 else 1.0


def train_one_target_coral(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    K: int,
):
    """
    Train CORAL (K-1 binary LGBM models) and tune decode threshold.
    Returns:
      boosters: list[lgb.Booster]
      best: dict
      train_pct: np.ndarray
      thr_grid_rows: list[dict]
      threshold_diag: list[dict]
    """
    Yt = make_coral_targets(y_tr, K)
    Yv = make_coral_targets(y_va, K)

    boosters = []
    threshold_diag = []

    for k in range(K - 1):
        yk_tr = Yt[:, k]
        yk_va = Yv[:, k]

        pos = float(yk_tr.sum())
        neg = float(len(yk_tr) - pos)

        spw_base = compute_spw(pos, neg, cap=WEIGHT_CFG["cap"], power=WEIGHT_CFG["power"])
        spw = spw_base * tail_weight(k, K, gamma=WEIGHT_CFG["gamma"])

        params = dict(LGB_BASE_PARAMS)
        params.update({
            "objective": "binary",
            "metric": "binary_logloss",
            "scale_pos_weight": spw,
        })

        dtrain = lgb.Dataset(X_tr, label=yk_tr, free_raw_data=False)
        dvalid = lgb.Dataset(X_va, label=yk_va, reference=dtrain, free_raw_data=False)

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        boosters.append(booster)

        threshold_diag.append({
            "k": k,
            "task": f"y > {k}",
            "pos": int(pos),
            "neg": int(neg),
            "scale_pos_weight": float(spw),
            "best_iteration": int(booster.best_iteration or 0),
        })

    # Tune decode threshold on validation
    proba_gt_va = np.vstack([b.predict(X_va, num_iteration=b.best_iteration) for b in boosters]).T
    train_pct = dist_pct(y_tr, K)

    best = None
    thr_grid_rows = []
    for t in DECODE_GRID:
        y_hat = decode_coral(proba_gt_va, t)
        mae = float(mean_absolute_error(y_va, y_hat))
        pred_pct = dist_pct(y_hat, K)
        drift = float(np.sum(np.abs(pred_pct - train_pct)))
        score = float(ALPHA * mae + (1.0 - ALPHA) * drift)

        row = {"threshold": float(t), "mae": mae, "drift_l1": drift, "score": score}
        thr_grid_rows.append(row)
        if best is None or score < best["score"]:
            best = row

    return boosters, best, train_pct, thr_grid_rows, threshold_diag


def main():
    # -------------------------
    # Load & clean training data
    # -------------------------
    df = pd.read_csv(TRAIN_CSV)

    # Clean ordinal targets first
    df = clean_ordinal_targets(df, TARGETS)
    assert_integer_targets(df, TARGETS)

    # Build feature columns
    exclude = set(TARGETS + ([ID_COL] if ID_COL and ID_COL in df.columns else []))
    feature_cols = [c for c in df.columns if c not in exclude]

    # Fix feature dtypes
    df, feature_cols, dtype_report = fix_feature_dtypes(df, feature_cols)
    assert_numeric_features(df, feature_cols)

    # Save dtype report (audit/debug)
    with open(RUN_REPORT_PATH, "w") as f:
        json.dump(dtype_report, f, indent=2)

    # Save Stage-0 feature list (used by Stage-3)
    joblib.dump({"selected_features": feature_cols}, STAGE0_PATH)

    X = df[feature_cols]

    # 80/20 split
    idx = np.arange(len(df))
    idx_tr, idx_va = train_test_split(idx, test_size=0.20, random_state=42, shuffle=True)
    X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]

    # -------------------------
    # Train per target
    # -------------------------
    for tgt in TARGETS:
        y_tr = df.iloc[idx_tr][tgt].astype(int).values
        y_va = df.iloc[idx_va][tgt].astype(int).values

        # assumes labels are 0..K-1
        K = int(max(df[tgt].max() + 1, 2))

        boosters, best, train_pct, thr_grid_rows, threshold_diag = train_one_target_coral(
            X_tr=X_tr, y_tr=y_tr,
            X_va=X_va, y_va=y_va,
            K=K,
        )

        artifact = {
            "target": tgt,
            "n_classes": K,
            "decode_threshold": float(best["threshold"]),
            "train_distribution_pct": train_pct.tolist(),

            # boosters only (no custom class pickle issues)
            "model_boosters": boosters,

            # traceability
            "tuning_alpha": float(ALPHA),
            "weight_cfg": WEIGHT_CFG,
            "decode_tuning_best": best,
            "decode_tuning_grid": thr_grid_rows,
            "threshold_diagnostics": threshold_diag,
            "feature_cols": feature_cols,
        }

        out_path = os.path.join(MODEL_DIR, f"{tgt}_coral_artifact.joblib")
        joblib.dump(artifact, out_path)

        # summary json (fast glance)
        with open(os.path.join(MODEL_DIR, f"{tgt}_training_summary.json"), "w") as f:
            json.dump(
                {
                    "target": tgt,
                    "n_classes": K,
                    "decode_threshold": artifact["decode_threshold"],
                    "mae": best["mae"],
                    "drift_l1": best["drift_l1"],
                    "score": best["score"],
                    "weight_cfg": WEIGHT_CFG,
                },
                f,
                indent=2,
            )

        print(
            f"[{tgt}] saved {out_path} | "
            f"thr={best['threshold']:.2f} | MAE={best['mae']:.3f} | drift={best['drift_l1']:.3f}"
        )


if __name__ == "__main__":
    main()



# coral_score_unseen.py
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dtype_utils import fix_feature_dtypes, assert_numeric_features

# =============================
# CONFIG (EDIT HERE)
# =============================
UNSEEN_CSV = "unseen_data.csv"
TARGETS = ["y_target1", "y_target2", "y_target3"]   # same list as training
ID_COL = "record_id"                                # optional, or None

ARTIFACT_DIR = "artifacts"
MODEL_DIR = f"{ARTIFACT_DIR}/coral_models"
STAGE0_PATH = f"{ARTIFACT_DIR}/stage0_features.joblib"

OUT_DIR = "unseen_reports"
DTYPE_REPORT_PATH = f"{OUT_DIR}/dtype_report_unseen.json"
COMBINED_PDF = f"{OUT_DIR}/business_report_unseen.pdf"

os.makedirs(OUT_DIR, exist_ok=True)


# =============================
# CORAL helpers
# =============================
def decode_coral(proba_gt: np.ndarray, t: float) -> np.ndarray:
    return (proba_gt >= float(t)).sum(axis=1).astype(int)

def dist_pct(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    c = np.bincount(y, minlength=K).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.zeros(K, dtype=float)

def sanity_checks(train_pct, pred_pct, rare_threshold=0.02, collapse_ratio=0.2, spike_ratio=3.0, drift_l1_warn=0.20):
    drift_l1 = float(np.sum(np.abs(pred_pct - train_pct)))
    rare = np.where(train_pct <= rare_threshold)[0].tolist()
    collapsed, spiked = [], []
    for c in rare:
        tp, pp = float(train_pct[c]), float(pred_pct[c])
        if tp > 0 and pp < tp * collapse_ratio:
            collapsed.append(int(c))
        if tp > 0 and pp > tp * spike_ratio:
            spiked.append(int(c))
    top_class = int(np.argmax(pred_pct))
    top_pct = float(np.max(pred_pct))
    return {
        "drift_l1": drift_l1,
        "drift_l1_warn": drift_l1 >= drift_l1_warn,
        "rare_classes_by_train": rare,
        "rare_collapsed": collapsed,
        "rare_spiked": spiked,
        "rare_collapsed_warn": len(collapsed) > 0,
        "rare_spiked_warn": len(spiked) > 0,
        "top_class": top_class,
        "top_class_pct": top_pct,
        "top_class_dominates_warn": top_pct >= 0.90,
    }

def add_page(pdf, target, train_pct, pred_pct, checks):
    K = len(train_pct)
    classes = np.arange(K)
    delta = pred_pct - train_pct

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait-ish

    ax1 = plt.subplot(3, 1, 1)
    w = 0.4
    ax1.bar(classes - w/2, train_pct*100, width=w, label="Train %")
    ax1.bar(classes + w/2, pred_pct*100, width=w, label="Unseen Pred %")
    ax1.set_xticks(classes)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Percent")
    ax1.set_title(f"{target} — Train vs Unseen Predicted Distribution")
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.bar(classes, delta*100)
    ax2.set_xticks(classes)
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Delta (pp)")
    ax2.set_title("Unseen Pred % minus Train % (percentage points)")

    ax3 = plt.subplot(3, 1, 3)
    ax3.axis("off")
    lines = [
        f"Drift L1: {checks['drift_l1']:.3f} (warn={checks['drift_l1_warn']})",
        f"Rare classes by train (<=2%): {checks['rare_classes_by_train']}",
        f"Rare collapsed: {checks['rare_collapsed']} (warn={checks['rare_collapsed_warn']})",
        f"Rare spiked: {checks['rare_spiked']} (warn={checks['rare_spiked_warn']})",
        f"Top class: {checks['top_class']} at {checks['top_class_pct']*100:.1f}% (dominates warn={checks['top_class_dominates_warn']})",
    ]
    ax3.text(0.01, 0.95, "\n".join(lines), va="top", ha="left")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    # -------------------------
    # Load Stage-0 features
    # -------------------------
    stage0 = joblib.load(STAGE0_PATH)
    feature_cols = stage0["selected_features"]

    # -------------------------
    # Load unseen data
    # -------------------------
    df = pd.read_csv(UNSEEN_CSV)

    # Ensure all required feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Fix feature dtypes for those columns
    df, feature_cols, dtype_report = fix_feature_dtypes(df, feature_cols)
    assert_numeric_features(df, feature_cols)

    with open(DTYPE_REPORT_PATH, "w") as f:
        json.dump(dtype_report, f, indent=2)

    X = df[feature_cols]

    # -------------------------
    # Predict + export
    # -------------------------
    checks_rows = []

    with PdfPages(COMBINED_PDF) as pdf:
        for tgt in TARGETS:
            artifact_path = os.path.join(MODEL_DIR, f"{tgt}_coral_artifact.joblib")
            art = joblib.load(artifact_path)

            boosters = art["model_boosters"]
            K = int(art["n_classes"])
            thr = float(art["decode_threshold"])
            train_pct = np.array(art["train_distribution_pct"], dtype=float)

            # proba_gt: (n_samples, K-1)
            proba_gt = np.vstack(
                [b.predict(X, num_iteration=b.best_iteration) for b in boosters]
            ).T

            y_pred = decode_coral(proba_gt, thr)
            pred_pct = dist_pct(y_pred, K)
            checks = sanity_checks(train_pct, pred_pct)

            # per-target prediction CSV
            pred_df = pd.DataFrame({f"{tgt}_pred": y_pred})
            if ID_COL and ID_COL in df.columns:
                pred_df.insert(0, ID_COL, df[ID_COL].values)
            pred_df.to_csv(os.path.join(OUT_DIR, f"{tgt}_unseen_predictions.csv"), index=False)

            # per-target distribution CSV
            dist_df = pd.DataFrame({
                "class": np.arange(K),
                "train_pct": train_pct,
                "unseen_pred_pct": pred_pct,
                "delta_pct": pred_pct - train_pct,
            })
            dist_df.to_csv(os.path.join(OUT_DIR, f"{tgt}_distribution_compare.csv"), index=False)

            # add page to combined PDF
            add_page(pdf, tgt, train_pct, pred_pct, checks)

            checks_rows.append({
                "target": tgt,
                "decode_threshold": thr,
                "drift_l1": checks["drift_l1"],
                "drift_l1_warn": checks["drift_l1_warn"],
                "rare_collapsed_warn": checks["rare_collapsed_warn"],
                "rare_spiked_warn": checks["rare_spiked_warn"],
                "top_class": checks["top_class"],
                "top_class_pct": checks["top_class_pct"],
                "top_class_dominates_warn": checks["top_class_dominates_warn"],
            })

    pd.DataFrame(checks_rows).to_csv(os.path.join(OUT_DIR, "combined_sanity_checks.csv"), index=False)
    print("Saved combined PDF:", COMBINED_PDF)
    print("Saved outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
