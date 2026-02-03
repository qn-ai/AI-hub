

import pandas as pd
import numpy as np

def fix_bad_dtypes_for_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    onehot_low_card: bool = False,
    low_card_max_unique: int = 20,
    high_card_drop_threshold: int = 200,
    report_top: int = 10,
):
    """
    Fix 'object' columns inside feature_cols:
      - If numeric-like: coerce to numeric
      - Else:
          - optional: one-hot if low-cardinality
          - otherwise: drop (safe)
    Returns:
      df_fixed, new_feature_cols, report_dict
    """
    df = df.copy()

    # Identify object feature columns
    obj_cols = [c for c in feature_cols if c in df.columns and df[c].dtype == "object"]

    def numeric_like_ratio(s: pd.Series, sample_size: int = 2000) -> float:
        x = s.dropna().astype(str).head(sample_size)

        if len(x) == 0:
            return 0.0

        # remove commas and spaces
        x = x.str.replace(",", "", regex=False).str.strip()

        # numeric pattern: int or float with optional leading sign
        return float(x.str.match(r"^-?\d+(\.\d+)?$", na=False).mean())

    numeric_like = []
    non_numeric = []

    for c in obj_cols:
        r = numeric_like_ratio(df[c])
        if r >= 0.80:
            numeric_like.append(c)
        else:
            non_numeric.append(c)

    # 1) Coerce numeric-like strings -> numeric
    for c in numeric_like:
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) Handle non-numeric object columns
    low_card = []
    high_card = []

    for c in non_numeric:
        nunq = df[c].nunique(dropna=True)
        if nunq <= low_card_max_unique:
            low_card.append(c)
        else:
            high_card.append(c)

    # Safe default: drop high-cardinality text/IDs
    drop_cols = list(high_card)

    # Optional: one-hot low-card categoricals
    if onehot_low_card and len(low_card) > 0:
        df = pd.get_dummies(df, columns=low_card, dummy_na=True)
        # feature cols update: remove original low-card col; add dummy cols
        new_feature_cols = []
        for c in feature_cols:
            if c in low_card:
                # replaced by dummy columns that start with c + "_"
                dummies = [dc for dc in df.columns if dc.startswith(c + "_")]
                new_feature_cols.extend(dummies)
            elif c in drop_cols:
                continue
            else:
                new_feature_cols.append(c)
    else:
        # drop all remaining non-numeric object cols (both low + high)
        drop_cols = drop_cols + low_card
        new_feature_cols = [c for c in feature_cols if c not in drop_cols]

    # Drop chosen columns if they exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 3) Final safety: force remaining features to numeric if possible
    # (anything still object is removed)
    still_obj = [c for c in new_feature_cols if c in df.columns and df[c].dtype == "object"]
    if still_obj:
        # Try coercion, then drop if still bad
        for c in still_obj:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        still_obj2 = [c for c in new_feature_cols if c in df.columns and df[c].dtype == "object"]
        if still_obj2:
            df = df.drop(columns=still_obj2)
            new_feature_cols = [c for c in new_feature_cols if c not in still_obj2]

    # 4) Replace inf and fill NaN (LightGBM-friendly)
    df[new_feature_cols] = df[new_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Report
    rep = {
        "object_feature_cols": obj_cols,
        "numeric_like_converted": numeric_like,
        "non_numeric_object_cols": non_numeric,
        "low_card_categoricals": low_card,
        "high_card_text_or_ids": high_card,
        "dropped_cols": drop_cols,
        "still_object_after_fix": [c for c in new_feature_cols if df[c].dtype == "object"],
    }

    # Quick print summary
    print("\n[dtype-fix] object features:", len(obj_cols))
    print("[dtype-fix] numeric-like converted:", len(numeric_like))
    print("[dtype-fix] non-numeric:", len(non_numeric))
    print("[dtype-fix] dropped:", len(drop_cols))
    if rep["still_object_after_fix"]:
        print("[dtype-fix] WARNING: still object after fix:", rep["still_object_after_fix"][:report_top])

    return df, new_feature_cols, rep


# df already loaded
# feature_cols already defined

df, feature_cols, dtype_report = fix_bad_dtypes_for_features(
    df,
    feature_cols,
    onehot_low_card=False,          # safest default (drop non-numeric)
    low_card_max_unique=20,
    high_card_drop_threshold=200,
)

# now build X using the cleaned df and updated feature_cols
X = df[feature_cols]


stage0_art = joblib.load(config.STAGE0_PATH)
feature_cols = stage0_art["selected_features"]

df_unseen = pd.read_csv(config.UNSEEN_CSV)

# Make sure unseen has the required columns (missing -> 0)
for c in feature_cols:
    if c not in df_unseen.columns:
        df_unseen[c] = 0.0

df_unseen, feature_cols, dtype_report = fix_bad_dtypes_for_features(
    df_unseen,
    feature_cols,
    onehot_low_card=False,
)

X_unseen = df_unseen[feature_cols]




# config.py

TRAIN_CSV = "training_data.csv"
UNSEEN_CSV = "unseen_data.csv"

TARGETS = ["y_target1", "y_target2", "y_target3"]   # <-- EDIT
ID_COL = "record_id"  # optional (set None if not present)

ARTIFACTS_DIR = "artifacts"
STAGE0_PATH = f"{ARTIFACTS_DIR}/stage0_features.joblib"
STAGE0_REPORT_DIR = f"{ARTIFACTS_DIR}/stage0_reports"
MODELS_DIR = f"{ARTIFACTS_DIR}/coral_models"

UNSEEN_REPORT_DIR = "unseen_reports"
COMBINED_PDF_NAME = "business_report_unseen.pdf"

# Stage-0 settings
MISSING_THRESHOLD = 0.80
CORR_THRESHOLD = 0.95
MAX_CORR_FEATURES = 400

# Threshold decode tuning
DECODE_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
ALPHA = 0.7  # MAE vs distribution realism (0.7 is a good business-safe default)

# Weight tuning grid
WEIGHT_GRID = [
    {"power": 0.4, "cap": 10.0, "gamma": 0.2},
    {"power": 0.5, "cap": 10.0, "gamma": 0.2},
    {"power": 0.5, "cap": 12.0, "gamma": 0.3},  # default-ish
    {"power": 0.6, "cap": 12.0, "gamma": 0.3},
    {"power": 0.5, "cap": 15.0, "gamma": 0.4},
]

# LightGBM base params
LGB_BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
}


# coral_train.py
import os, json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import config


# =============================
# Stage-0: feature filter (embedded)
# =============================
def stage0_fit_transform(X_tr: pd.DataFrame, y_tr_for_importance: np.ndarray):
    # 1) drop high-missing
    miss_rate = X_tr.isna().mean(axis=0)
    dropped_missing = miss_rate[miss_rate >= config.MISSING_THRESHOLD].index.tolist()
    kept = [c for c in X_tr.columns if c not in set(dropped_missing)]
    X_tr1 = X_tr[kept]

    # 2) quick LGBM importance (cheap)
    X_imp = X_tr1.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_imp = np.asarray(y_tr_for_importance).astype(int)

    params = {
        "objective": "multiclass" if len(np.unique(y_imp)) > 2 else "binary",
        "metric": "multi_logloss" if len(np.unique(y_imp)) > 2 else "binary_logloss",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
    }
    if params["objective"] == "multiclass":
        params["num_class"] = int(np.max(y_imp) + 1)

    booster = lgb.train(
        params=params,
        train_set=lgb.Dataset(X_imp, label=y_imp, free_raw_data=False),
        num_boost_round=300,
    )
    gain = booster.feature_importance(importance_type="gain")
    feats = booster.feature_name()
    imp = {f: float(g) for f, g in zip(feats, gain)}
    for f in X_tr1.columns:
        imp.setdefault(f, 0.0)

    # 3) correlation prune (spearman abs >= threshold), keep higher-importance
    pool = kept
    if config.MAX_CORR_FEATURES is not None and len(pool) > config.MAX_CORR_FEATURES:
        pool = sorted(pool, key=lambda f: imp.get(f, 0.0), reverse=True)[: config.MAX_CORR_FEATURES]

    Xc = X_tr1[pool].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = Xc.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    corr_pairs = []
    cols_sorted = sorted(list(upper.columns), key=lambda f: imp.get(f, 0.0), reverse=True)

    for col in cols_sorted:
        if col in to_drop:
            continue
        high = upper[col][upper[col] >= config.CORR_THRESHOLD].dropna()
        for other, val in high.items():
            if other in to_drop:
                continue
            # drop lower-importance one
            if imp.get(col, 0.0) >= imp.get(other, 0.0):
                kept_f, drop_f = col, other
            else:
                kept_f, drop_f = other, col
            to_drop.add(drop_f)
            corr_pairs.append((kept_f, drop_f, float(val)))

    dropped_corr = sorted(list(to_drop))
    selected_features = [f for f in kept if f not in to_drop]

    artifacts = {
        "selected_features": selected_features,
        "dropped_missing": dropped_missing,
        "dropped_corr": dropped_corr,
        "corr_pairs_dropped": corr_pairs,
        "feature_importance_gain": imp,
        "config": {
            "missing_threshold": config.MISSING_THRESHOLD,
            "corr_threshold": config.CORR_THRESHOLD,
            "max_corr_features": config.MAX_CORR_FEATURES,
        },
    }
    return artifacts


# =============================
# CORAL core + improved weights
# =============================
def make_coral_targets(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    thr = np.arange(K - 1)
    return (y[:, None] > thr[None, :]).astype(np.int8)

def decode_coral(proba_gt: np.ndarray, t: float) -> np.ndarray:
    return (proba_gt >= t).sum(axis=1).astype(int)

def dist_pct(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    c = np.bincount(y, minlength=K).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.zeros(K)

def compute_spw(pos: float, neg: float, cap: float, power: float) -> float:
    if pos <= 0:
        return 1.0
    return float(min((neg / pos) ** power, cap))

def tail_weight(k: int, K: int, gamma: float) -> float:
    return float(1.0 + gamma * (k / (K - 1))) if K > 1 else 1.0


class LGBMCoralModel:
    def __init__(self, models, n_classes: int):
        self.models = models
        self.n_classes = n_classes

    def predict_proba_gt(self, X) -> np.ndarray:
        return np.vstack([m.predict(X, num_iteration=m.best_iteration) for m in self.models]).T

    def predict(self, X, threshold: float) -> np.ndarray:
        return decode_coral(self.predict_proba_gt(X), threshold)


def train_coral_with_weights(X_tr, y_tr, X_va, y_va, K, base_params, weight_cfg):
    Yt = make_coral_targets(y_tr, K)
    Yv = make_coral_targets(y_va, K)

    models = []
    diag = []

    for k in range(K - 1):
        yk_tr = Yt[:, k]
        dtrain = lgb.Dataset(X_tr, label=yk_tr, free_raw_data=False)
        dvalid = lgb.Dataset(X_va, label=Yv[:, k], reference=dtrain, free_raw_data=False)

        pos = float(yk_tr.sum())
        neg = float(len(yk_tr) - pos)

        spw_base = compute_spw(pos, neg, cap=weight_cfg["cap"], power=weight_cfg["power"])
        spw = spw_base * tail_weight(k, K, gamma=weight_cfg["gamma"])

        params = dict(base_params)
        params.update({
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "scale_pos_weight": spw,
        })

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )

        models.append(booster)
        diag.append({
            "k": k,
            "task": f"y > {k}",
            "pos": int(pos),
            "neg": int(neg),
            "spw_used": float(spw),
            "best_iter": int(booster.best_iteration or 0),
        })

    return LGBMCoralModel(models, K), diag


def tune_decode_threshold(proba_gt_va, y_va, K, train_pct, grid, alpha):
    best = None
    rows = []
    for t in grid:
        t = float(t)
        y_hat = decode_coral(proba_gt_va, t)
        mae = float(mean_absolute_error(y_va, y_hat))
        pred_pct = dist_pct(y_hat, K)
        drift = float(np.sum(np.abs(pred_pct - train_pct)))
        score = alpha * mae + (1 - alpha) * drift
        row = {"threshold": t, "mae": mae, "drift_l1": drift, "score": score}
        rows.append(row)
        if best is None or score < best["score"]:
            best = row
    return best, rows


def main():
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(config.STAGE0_REPORT_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    df = pd.read_csv(config.TRAIN_CSV)

    exclude = set(config.TARGETS + ([config.ID_COL] if config.ID_COL and config.ID_COL in df.columns else []))
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 80/20 split
    idx = np.arange(len(df))
    idx_tr, idx_va = train_test_split(idx, test_size=0.20, random_state=42, shuffle=True)
    X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]

    # Stage-0 fit on train only (use first target for importance ranking)
    stage0_art = stage0_fit_transform(X_tr, df.iloc[idx_tr][config.TARGETS[0]].astype(int).values)

    # Save Stage-0 artifact + reports
    joblib.dump(stage0_art, config.STAGE0_PATH)
    pd.Series(stage0_art["selected_features"]).to_csv(f"{config.STAGE0_REPORT_DIR}/selected_features.csv", index=False)
    pd.Series(stage0_art["dropped_missing"]).to_csv(f"{config.STAGE0_REPORT_DIR}/dropped_missing.csv", index=False)
    pd.Series(stage0_art["dropped_corr"]).to_csv(f"{config.STAGE0_REPORT_DIR}/dropped_corr.csv", index=False)
    pd.DataFrame(stage0_art["corr_pairs_dropped"], columns=["kept", "dropped", "abs_corr"]).to_csv(
        f"{config.STAGE0_REPORT_DIR}/corr_pairs_dropped.csv", index=False
    )

    sel = stage0_art["selected_features"]
    X_tr_f = X_tr[sel].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_va_f = X_va[sel].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Train per target with weight tuning + threshold tuning
    for t in config.TARGETS:
        y_tr = df.iloc[idx_tr][t].astype(int).values
        y_va = df.iloc[idx_va][t].astype(int).values
        K = int(np.max(df[t].dropna().astype(int).values) + 1)

        candidates = []
        for wcfg in config.WEIGHT_GRID:
            coral, diag = train_coral_with_weights(
                X_tr_f, y_tr, X_va_f, y_va, K, config.LGB_BASE_PARAMS, wcfg
            )
            train_pct = dist_pct(y_tr, K)
            proba_gt_va = coral.predict_proba_gt(X_va_f)
            best_thr, thr_grid = tune_decode_threshold(
                proba_gt_va, y_va, K, train_pct, config.DECODE_GRID, config.ALPHA
            )
            candidates.append({
                "weight_cfg": wcfg,
                "coral": coral,
                "diag": diag,
                "train_pct": train_pct,
                "best_thr": best_thr,
                "thr_grid": thr_grid,
            })

        candidates.sort(key=lambda d: d["best_thr"]["score"])
        best = candidates[0]

        artifact = {
            "target": t,
            "n_classes": K,
            "decode_threshold": float(best["best_thr"]["threshold"]),
            "train_distribution_pct": best["train_pct"].tolist(),
            "tuning_best": best["best_thr"],
            "tuning_grid": best["thr_grid"],
            "weight_cfg_best": best["weight_cfg"],
            "threshold_diagnostics": best["diag"],
            "feature_cols": sel,
            "model": best["coral"],
        }

        out_path = f"{config.MODELS_DIR}/{t}_coral_artifact.joblib"
        joblib.dump(artifact, out_path)

        with open(f"{config.MODELS_DIR}/{t}_training_summary.json", "w") as f:
            json.dump(
                {
                    "target": t,
                    "K": K,
                    "decode_threshold": artifact["decode_threshold"],
                    "best_weight_cfg": artifact["weight_cfg_best"],
                    "best_mae": artifact["tuning_best"]["mae"],
                    "best_drift_l1": artifact["tuning_best"]["drift_l1"],
                    "best_score": artifact["tuning_best"]["score"],
                },
                f,
                indent=2,
            )

        print(
            f"[{t}] saved {out_path} | weights={artifact['weight_cfg_best']} | "
            f"thr={artifact['decode_threshold']:.2f} | mae={artifact['tuning_best']['mae']:.3f} | "
            f"drift={artifact['tuning_best']['drift_l1']:.3f}"
        )


if __name__ == "__main__":
    main()


# coral_score_unseen.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config


def dist_pct(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    c = np.bincount(y, minlength=K).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.zeros(K)

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

    fig = plt.figure(figsize=(8.27, 11.69))
    ax1 = plt.subplot(3, 1, 1)
    w = 0.4
    ax1.bar(classes - w/2, train_pct*100, width=w, label="Train %")
    ax1.bar(classes + w/2, pred_pct*100, width=w, label="Unseen Pred %")
    ax1.set_xticks(classes); ax1.set_xlabel("Class"); ax1.set_ylabel("Percent")
    ax1.set_title(f"{target} — Train vs Unseen Predicted Distribution")
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.bar(classes, delta*100)
    ax2.set_xticks(classes); ax2.set_xlabel("Class"); ax2.set_ylabel("Delta (pp)")
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
    os.makedirs(config.UNSEEN_REPORT_DIR, exist_ok=True)

    stage0_art = joblib.load(config.STAGE0_PATH)
    feature_cols = stage0_art["selected_features"]

    df_unseen = pd.read_csv(config.UNSEEN_CSV)

    # Build X_unseen with required columns
    X_unseen = df_unseen.copy()
    for c in feature_cols:
        if c not in X_unseen.columns:
            X_unseen[c] = 0.0
    X_unseen = X_unseen[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    combined_pdf = os.path.join(config.UNSEEN_REPORT_DIR, config.COMBINED_PDF_NAME)
    checks_rows = []

    with PdfPages(combined_pdf) as pdf:
        for t in config.TARGETS:
            art = joblib.load(f"{config.MODELS_DIR}/{t}_coral_artifact.joblib")

            K = art["n_classes"]
            thr = art["decode_threshold"]
            train_pct = np.array(art["train_distribution_pct"], dtype=float)

            coral = art["model"]
            y_pred = coral.predict(X_unseen, threshold=thr)
            pred_pct = dist_pct(y_pred, K)
            checks = sanity_checks(train_pct, pred_pct)

            # per-target prediction CSV
            pred_df = pd.DataFrame({f"{t}_pred": y_pred})
            if config.ID_COL and config.ID_COL in df_unseen.columns:
                pred_df.insert(0, config.ID_COL, df_unseen[config.ID_COL].values)
            pred_df.to_csv(os.path.join(config.UNSEEN_REPORT_DIR, f"{t}_unseen_predictions.csv"), index=False)

            # per-target distribution CSV
            dist_df = pd.DataFrame({
                "class": np.arange(K),
                "train_pct": train_pct,
                "unseen_pred_pct": pred_pct,
                "delta_pct": pred_pct - train_pct,
            })
            dist_df.to_csv(os.path.join(config.UNSEEN_REPORT_DIR, f"{t}_distribution_compare.csv"), index=False)

            # pdf page
            add_page(pdf, t, train_pct, pred_pct, checks)

            checks_rows.append({"target": t, "decode_threshold": thr, "best_weight_cfg": art.get("weight_cfg_best", {}), **checks})

    pd.DataFrame(checks_rows).to_csv(os.path.join(config.UNSEEN_REPORT_DIR, "combined_sanity_checks.csv"), index=False)
    print("Saved:", combined_pdf)
    print("Saved per-target CSVs in:", config.UNSEEN_REPORT_DIR)


if __name__ == "__main__":
    main()
