# stage0_feature_filter.py
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

@dataclass
class Stage0Config:
    missing_threshold: float = 0.80
    corr_threshold: float = 0.95
    corr_method: str = "spearman"
    max_corr_features: Optional[int] = 400
    importance_num_boost_round: int = 300
    importance_early_stopping: int = 50
    random_state: int = 42

@dataclass
class Stage0Artifacts:
    selected_features: List[str]
    dropped_missing: List[str]
    dropped_corr: List[str]
    corr_pairs_dropped: List[Tuple[str, str, float]]
    feature_importance_gain: Dict[str, float]
    config: Dict[str, Any]

class Stage0FeatureFilter:
    def __init__(self, config: Stage0Config = Stage0Config()):
        self.config = config
        self.artifacts: Optional[Stage0Artifacts] = None

    def fit(self, X_train: pd.DataFrame, y_train, X_valid=None, y_valid=None) -> "Stage0FeatureFilter":
        X_train = self._ensure_df(X_train)
        y_train = np.asarray(y_train)

        kept, dropped_missing = self._drop_high_missing(X_train)
        X_tr = X_train[kept]

        gain_imp = self._quick_lgbm_importance(X_tr, y_train)

        pool = kept
        if self.config.max_corr_features is not None and len(pool) > self.config.max_corr_features:
            pool = sorted(pool, key=lambda f: gain_imp.get(f, 0.0), reverse=True)[: self.config.max_corr_features]

        kept_final, dropped_corr, pairs = self._correlation_prune(X_tr[pool], kept, gain_imp)

        self.artifacts = Stage0Artifacts(
            selected_features=kept_final,
            dropped_missing=dropped_missing,
            dropped_corr=dropped_corr,
            corr_pairs_dropped=pairs,
            feature_importance_gain=gain_imp,
            config=asdict(self.config),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X = self._ensure_df(X)

        missing_cols = [c for c in self.artifacts.selected_features if c not in X.columns]
        if missing_cols:
            X = X.copy()
            for c in missing_cols:
                X[c] = 0.0

        return X[self.artifacts.selected_features]

    def save(self, path: str) -> None:
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"artifacts": self.artifacts}, path)

    @classmethod
    def load(cls, path: str) -> "Stage0FeatureFilter":
        payload = joblib.load(path)
        obj = cls(config=Stage0Config(**payload["artifacts"].config))
        obj.artifacts = payload["artifacts"]
        return obj

    def export_report(self, out_dir: str, prefix: str = "stage0") -> None:
        self._check_fitted()
        os.makedirs(out_dir, exist_ok=True)

        pd.Series(self.artifacts.selected_features, name="feature").to_csv(
            os.path.join(out_dir, f"{prefix}_selected_features.csv"), index=False
        )
        pd.Series(self.artifacts.dropped_missing, name="feature").to_csv(
            os.path.join(out_dir, f"{prefix}_dropped_missing.csv"), index=False
        )
        pd.Series(self.artifacts.dropped_corr, name="feature").to_csv(
            os.path.join(out_dir, f"{prefix}_dropped_corr.csv"), index=False
        )
        pd.DataFrame(self.artifacts.corr_pairs_dropped, columns=["kept", "dropped", "abs_corr"]).to_csv(
            os.path.join(out_dir, f"{prefix}_corr_pairs_dropped.csv"), index=False
        )
        (pd.Series(self.artifacts.feature_importance_gain, name="gain_importance")
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"index": "feature"})
           .to_csv(os.path.join(out_dir, f"{prefix}_feature_importance_gain.csv"), index=False))

    @staticmethod
    def _ensure_df(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _check_fitted(self):
        if self.artifacts is None:
            raise RuntimeError("Stage0FeatureFilter not fitted. Call fit() first.")

    def _drop_high_missing(self, X: pd.DataFrame):
        miss_rate = X.isna().mean(axis=0)
        dropped = miss_rate[miss_rate >= self.config.missing_threshold].index.tolist()
        kept = [c for c in X.columns if c not in set(dropped)]
        return kept, dropped

    def _quick_lgbm_importance(self, X_train: pd.DataFrame, y_train: np.ndarray):
        X_tr = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_tr = y_train

        uniq = np.unique(y_tr[~pd.isna(y_tr)])
        is_binary = len(uniq) <= 2

        if is_binary:
            params = {"objective":"binary","metric":"binary_logloss","learning_rate":0.05,
                      "num_leaves":63,"min_data_in_leaf":50,"feature_fraction":0.8,
                      "bagging_fraction":0.8,"bagging_freq":1,"lambda_l2":1.0,
                      "verbosity":-1,"seed":self.config.random_state}
        else:
            n_classes = int(np.max(uniq) + 1)
            params = {"objective":"multiclass","num_class":n_classes,"metric":"multi_logloss",
                      "learning_rate":0.05,"num_leaves":63,"min_data_in_leaf":50,
                      "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":1,
                      "lambda_l2":1.0,"verbosity":-1,"seed":self.config.random_state}

        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        booster = lgb.train(params=params, train_set=dtrain, num_boost_round=self.config.importance_num_boost_round)

        gain = booster.feature_importance(importance_type="gain")
        feats = booster.feature_name()
        imp = {f: float(g) for f, g in zip(feats, gain)}
        for f in X_train.columns:
            imp.setdefault(f, 0.0)
        return imp

    def _correlation_prune(self, X_pool: pd.DataFrame, all_features: List[str], gain_importance: Dict[str, float]):
        Xc = X_pool.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr = Xc.corr(method=self.config.corr_method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        pairs = []

        cols_sorted = sorted(list(upper.columns), key=lambda f: gain_importance.get(f, 0.0), reverse=True)
        for col in cols_sorted:
            if col in to_drop:
                continue
            high = upper[col][upper[col] >= self.config.corr_threshold].dropna()
            for other, val in high.items():
                if other in to_drop:
                    continue
                if gain_importance.get(col, 0.0) >= gain_importance.get(other, 0.0):
                    kept, dropped = col, other
                else:
                    kept, dropped = other, col
                to_drop.add(dropped)
                pairs.append((kept, dropped, float(val)))

        dropped_corr = sorted(list(to_drop))
        kept_final = [f for f in all_features if f not in to_drop]
        return kept_final, dropped_corr, pairs


### Stage 2
# coral_train.py
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from stage0_feature_filter import Stage0FeatureFilter, Stage0Config


def make_coral_targets(y: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    thr = np.arange(K - 1)
    return (y[:, None] > thr[None, :]).astype(np.int8)

def decode_coral(proba_gt: np.ndarray, t: float) -> np.ndarray:
    return (proba_gt >= t).sum(axis=1).astype(int)

@dataclass
class LGBMCoralModel:
    models: list
    n_classes: int

    def predict_proba_gt(self, X) -> np.ndarray:
        return np.vstack([m.predict(X, num_iteration=m.best_iteration) for m in self.models]).T

    def predict(self, X, threshold: float) -> np.ndarray:
        return decode_coral(self.predict_proba_gt(X), threshold)

def train_lgbm_coral(X_tr, y_tr, K, X_va, y_va, base_params):
    Yt = make_coral_targets(y_tr, K)
    Yv = make_coral_targets(y_va, K)

    models = []
    for k in range(K - 1):
        yk_tr = Yt[:, k]
        dtrain = lgb.Dataset(X_tr, label=yk_tr, free_raw_data=False)

        pos = float(yk_tr.sum())
        neg = float(len(yk_tr) - pos)
        spw = (neg / pos) if pos > 0 else 1.0

        params = dict(base_params)
        params["scale_pos_weight"] = spw
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
        params["verbosity"] = -1

        dvalid = lgb.Dataset(X_va, label=Yv[:, k], reference=dtrain, free_raw_data=False)

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        models.append(booster)

    return LGBMCoralModel(models=models, n_classes=K)

def dist_pct(y, K):
    c = np.bincount(np.asarray(y).astype(int), minlength=K).astype(float)
    s = c.sum()
    return c / s if s > 0 else np.zeros(K)

def tune_threshold(proba_gt_va, y_va, K, train_pct, grid=np.linspace(0.3,0.7,17), alpha=0.7):
    best = None
    rows = []
    for t in grid:
        y_hat = decode_coral(proba_gt_va, float(t))
        mae = float(mean_absolute_error(y_va, y_hat))
        pred_pct = dist_pct(y_hat, K)
        drift = float(np.sum(np.abs(pred_pct - train_pct)))
        score = alpha * mae + (1 - alpha) * drift
        rows.append({"threshold": float(t), "mae": mae, "drift_l1": drift, "score": score})
        if best is None or score < best["score"]:
            best = {"threshold": float(t), "mae": mae, "drift_l1": drift, "score": score}
    return best, rows

def main():
    TRAIN_CSV = "training_data.csv"
    TARGETS = ["y_target1", "y_target2", "y_target3"]  # <-- edit
    ID_COL = "record_id"  # optional (doesn't affect training)

    OUT_STAGE0 = "artifacts/stage0_features.joblib"
    OUT_STAGE0_REPORT = "artifacts/stage0_reports"
    OUT_MODELS = "artifacts/coral_models"
    os.makedirs(OUT_MODELS, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)

    feature_cols = [c for c in df.columns if c not in TARGETS + ([ID_COL] if ID_COL in df.columns else [])]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 80/20 split
    idx = np.arange(len(df))
    idx_tr, idx_va = train_test_split(idx, test_size=0.20, random_state=42, shuffle=True)
    X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]

    # Stage-0 fit on train only (use first target for importance ranking)
    stage0 = Stage0FeatureFilter(Stage0Config(missing_threshold=0.80, corr_threshold=0.95, max_corr_features=400))
    stage0.fit(X_tr, df.iloc[idx_tr][TARGETS[0]].astype(int).values)
    X_tr_f = stage0.transform(X_tr)
    X_va_f = stage0.transform(X_va)
    stage0.save(OUT_STAGE0)
    stage0.export_report(OUT_STAGE0_REPORT)

    base_params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": 42,
    }

    for t in TARGETS:
        y_tr = df.iloc[idx_tr][t].astype(int).values
        y_va = df.iloc[idx_va][t].astype(int).values

        K = int(np.max(df[t].dropna().astype(int).values) + 1)

        coral = train_lgbm_coral(X_tr_f, y_tr, K, X_va_f, y_va, base_params)
        train_pct = dist_pct(y_tr, K)

        proba_gt_va = coral.predict_proba_gt(X_va_f)
        best, grid_rows = tune_threshold(proba_gt_va, y_va, K, train_pct, alpha=0.7)

        artifact = {
            "target": t,
            "n_classes": K,
            "decode_threshold": best["threshold"],
            "train_distribution_pct": train_pct.tolist(),
            "tuning_best": best,
            "tuning_grid": grid_rows,
            "feature_cols": stage0.artifacts.selected_features,
            "model": coral,
        }

        path = os.path.join(OUT_MODELS, f"{t}_coral_artifact.joblib")
        joblib.dump(artifact, path)
        print(f"Saved {t} -> {path} (best_thr={best['threshold']:.2f}, mae={best['mae']:.3f}, drift={best['drift_l1']:.3f})")

if __name__ == "__main__":
    main()


### Stage 3

# coral_score_unseen.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stage0_feature_filter import Stage0FeatureFilter


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
    UNSEEN_CSV = "unseen_data.csv"
    TARGETS = ["y_target1", "y_target2", "y_target3"]   # <-- edit
    ID_COL = "record_id"  # optional

    STAGE0_PATH = "artifacts/stage0_features.joblib"
    MODELS_DIR = "artifacts/coral_models"
    OUT_DIR = "unseen_reports"
    os.makedirs(OUT_DIR, exist_ok=True)

    df_unseen = pd.read_csv(UNSEEN_CSV)

    # Use the exact feature set saved in Stage-0
    stage0 = Stage0FeatureFilter.load(STAGE0_PATH)
    feature_cols = stage0.artifacts.selected_features

    # Build X_unseen and ensure all required columns exist
    X_unseen = df_unseen.copy()
    for c in feature_cols:
        if c not in X_unseen.columns:
            X_unseen[c] = 0.0
    X_unseen = X_unseen[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    combined_pdf = os.path.join(OUT_DIR, "business_report_unseen.pdf")
    checks_rows = []

    with PdfPages(combined_pdf) as pdf:
        for t in TARGETS:
            artifact_path = os.path.join(MODELS_DIR, f"{t}_coral_artifact.joblib")
            art = joblib.load(artifact_path)

            K = art["n_classes"]
            thr = art["decode_threshold"]
            train_pct = np.array(art["train_distribution_pct"], dtype=float)

            coral = art["model"]
            y_pred = coral.predict(X_unseen, threshold=thr)
            pred_pct = dist_pct(y_pred, K)
            checks = sanity_checks(train_pct, pred_pct)

            # --- per-target prediction CSV ---
            pred_df = pd.DataFrame({f"{t}_pred": y_pred})
            if ID_COL in df_unseen.columns:
                pred_df.insert(0, ID_COL, df_unseen[ID_COL].values)
            pred_df.to_csv(os.path.join(OUT_DIR, f"{t}_unseen_predictions.csv"), index=False)

            # --- per-target distribution CSV ---
            dist_df = pd.DataFrame({
                "class": np.arange(K),
                "train_pct": train_pct,
                "unseen_pred_pct": pred_pct,
                "delta_pct": pred_pct - train_pct,
            })
            dist_df.to_csv(os.path.join(OUT_DIR, f"{t}_distribution_compare.csv"), index=False)

            # --- add PDF page ---
            add_page(pdf, t, train_pct, pred_pct, checks)

            checks_rows.append({"target": t, "decode_threshold": thr, **checks})

    pd.DataFrame(checks_rows).to_csv(os.path.join(OUT_DIR, "combined_sanity_checks.csv"), index=False)

    print("Saved combined PDF:", combined_pdf)
    print("Saved per-target CSVs in:", OUT_DIR)

if __name__ == "__main__":
    main()
