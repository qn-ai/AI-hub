Stage 0: feature cleaning
Stage 1: per-target CORAL training (temporary)
Stage 2: PCA diagnostics on training features
Stage 3: feature pruning decision
Stage 4: FINAL per-target CORAL training
Stage 5: predict on full_df

# config.py

# ---------------------------
# Core inputs/outputs
# ---------------------------
TRAIN_CSV = "training_data.csv"
UNSEEN_CSV = "unseen_data.csv"   # set None if no unseen

# If TARGETS is None, script auto-detects columns starting with TARGET_PREFIX ("y_")
TARGETS = None
TARGET_PREFIX = "y_"

# If ID_COL is None, script will auto-detect first column starting with ID_PREFIX ("id_")
ID_COL = None
ID_PREFIX = "id_"

FEATURE_PREFIX = "ft_"

ARTIFACTS_DIR = "artifacts"
MODELS_DIR = f"{ARTIFACTS_DIR}/coral_models"
OUT_DIR = f"{ARTIFACTS_DIR}/final_predictions"

# Optional: if you have a stage-0 selected feature artifact
STAGE0_PATH = f"{ARTIFACTS_DIR}/stage0_features.joblib"

# Optional: explicit feature list overrides everything
# FEATURE_COLS = ["ft_a", "ft_b", ...]

# ---------------------------
# Training guards
# ---------------------------
MIN_ROWS_PER_TARGET = 80

# Decode threshold candidates (tau grid)
DECODE_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# ---------------------------
# PCA Gate settings
# ---------------------------
PCA_GATE_DIR = f"{ARTIFACTS_DIR}/pca_gate"
PCA_N_COMPONENTS = 20
PCA_TOP_K_LOADINGS = 15
PCA_PLOT_MAX = 50

# Auto-drop rules
PCA_DROP_MISSING_FRAC = 0.80
PCA_DROP_DOMINANT_PC_COUNT = 3
PCA_DROP_LOW_VAR_QUANTILE = 0.00

# Never drop these (optional)
PCA_ALWAYS_KEEP = []

# ---------------------------
# Feature drop safety check (baseline LightGBM importances)
# ---------------------------
PCA_SAFETY_CHECK_ENABLED = True
PCA_SAFETY_TOP_N = 50

# With 200+ targets, increase coverage + thresholds
PCA_SAFETY_MAX_TARGETS = 50
PCA_SAFETY_MIN_ROWS = 300
PCA_SAFETY_SAMPLE_ROWS = 8000
PCA_SAFETY_N_ESTIMATORS = 400
PCA_SAFETY_RANDOM_STATE = 42

# Never drop features that appear in >= X targets' top-N importance lists
PCA_SAFETY_MIN_TARGET_SUPPORT = 10

# Weighted support rule
PCA_SAFETY_WEIGHTED_SUPPORT_ENABLED = True
PCA_SAFETY_WEIGHTED_SUPPORT_THRESHOLD = 400

# ---------------------------
# Prediction output controls
# ---------------------------
OUTPUT_PROB_COLUMNS = True
MAX_PROB_CLASSES = 10



# lgbm_coral.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except Exception as e:
    raise ImportError("lightgbm is required. Install with: pip install lightgbm") from e


# -----------------------------
# Helpers
# -----------------------------

def _to_numpy(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values
    if isinstance(X, pd.Series):
        return X.to_frame().values
    return np.asarray(X)

def _ensure_int_labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    if not np.issubdtype(y.dtype, np.integer):
        # allow float with .0
        if np.all(np.isfinite(y)) and np.all(np.equal(np.mod(y, 1), 0)):
            y = y.astype(int)
        else:
            raise ValueError("y must be integer-coded ordinal labels 0..K-1.")
    return y.astype(int)

def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

def _build_class_weights(y: np.ndarray, alpha: float) -> Dict[int, float]:
    """
    Inverse frequency weights with exponent alpha.
    alpha=0 -> all weights = 1
    alpha=1 -> standard inverse frequency
    """
    vals, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    inv = (1.0 / np.clip(freq, 1e-12, None)) ** alpha
    inv = inv / np.mean(inv)  # normalize around 1.0
    return {int(v): float(w) for v, w in zip(vals, inv)}

def _decode_coral(cum_probs: np.ndarray, tau: float) -> np.ndarray:
    """
    cum_probs shape: (n, K-1), column k-1 is P(y >= k)
    pred = number of cum_probs >= tau
    """
    return (cum_probs >= tau).sum(axis=1).astype(int)

def _has_min_count_per_class(y: np.ndarray, cv_folds: int) -> bool:
    """
    Enforce: every class has at least cv_folds samples.
    Needed for stable StratifiedKFold tuning (especially with QWK).
    """
    _, counts = np.unique(y, return_counts=True)
    return bool(np.all(counts >= cv_folds))


# -----------------------------
# Config
# -----------------------------

@dataclass
class LGBMCoralConfig:
    # Base LightGBM params
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int = 42

    # Early stopping
    early_stopping_rounds: int = 100
    eval_metric: str = "binary_logloss"

    # Weight tuning grids
    alpha_grid: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    pos_mult_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    tau_grid: Tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

    # CV tuning
    cv_folds: int = 5
    min_rows_per_fold: int = 100  # if too small, skip CV weight tuning


# -----------------------------
# Main class
# -----------------------------

class LGBMCoralModel:
    """
    CORAL-style ordinal model using K-1 binary LGBM models for cumulative probabilities.

    For K classes (0..K-1), train thresholds k=1..K-1:
      model_k learns P(y >= k)

    Weights:
      - base sample weight from original class weights: inv_freq^alpha
      - for each threshold model, positive samples (y>=k) get multiplied by pos_mult

    Tuning:
      - If every class count >= cv_folds, run CV grid over alpha, pos_mult, tau to maximize QWK
      - Otherwise, skip weight tuning and use defaults, but do decode-only tau tuning on a holdout set
    """

    def __init__(
        self,
        num_classes: int,
        config: Optional[LGBMCoralConfig] = None,
        decode_grid: Optional[List[float]] = None,  # alias for tau_grid
    ):
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)
        self.cfg = config or LGBMCoralConfig()
        if decode_grid is not None:
            self.cfg = LGBMCoralConfig(**{**asdict(self.cfg), "tau_grid": tuple(decode_grid)})

        self.models_: List[LGBMClassifier] = []
        self.best_params_: Dict[str, Any] = {}
        self.fitted_: bool = False

    def _make_base_estimator(self) -> LGBMClassifier:
        c = self.cfg
        return LGBMClassifier(
            n_estimators=c.n_estimators,
            learning_rate=c.learning_rate,
            num_leaves=c.num_leaves,
            min_child_samples=c.min_child_samples,
            subsample=c.subsample,
            colsample_bytree=c.colsample_bytree,
            reg_lambda=c.reg_lambda,
            n_jobs=c.n_jobs,
            random_state=c.random_state,
        )

    def _fit_one_threshold(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        k: int,
        class_w: Dict[int, float],
        pos_mult: float,
    ) -> LGBMClassifier:
        yb_tr = (y_tr >= k).astype(int)
        yb_va = (y_va >= k).astype(int)

        w_tr = np.array([class_w[int(yy)] for yy in y_tr], dtype=float)
        w_va = np.array([class_w[int(yy)] for yy in y_va], dtype=float)

        # boost positives for this threshold
        w_tr *= np.where(yb_tr == 1, pos_mult, 1.0)
        w_va *= np.where(yb_va == 1, pos_mult, 1.0)

        m = self._make_base_estimator()
        m.fit(
            X_tr,
            yb_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, yb_va)],
            eval_sample_weight=[w_va],
            eval_metric=self.cfg.eval_metric,
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
        )
        return m

    def _predict_cum_probs_with_models(self, X: np.ndarray, models: List[LGBMClassifier]) -> np.ndarray:
        probs = []
        for m in models:
            probs.append(m.predict_proba(X)[:, 1])  # P(y>=k)
        return np.vstack(probs).T  # (n, K-1)

    def _tune_tau_on_valid(self, y_valid: np.ndarray, cum_probs_valid: np.ndarray) -> Dict[str, float]:
        best_tau = None
        best_score = -np.inf
        for tau in self.cfg.tau_grid:
            y_pred = _decode_coral(cum_probs_valid, tau=float(tau))
            score = _qwk(y_valid, y_pred)
            if score > best_score:
                best_score = score
                best_tau = float(tau)
        return {"tau": float(best_tau), "tau_score_qwk": float(best_score)}

    def _cv_score_params(self, X: np.ndarray, y: np.ndarray, alpha: float, pos_mult: float, tau: float) -> float:
        skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
        class_w = _build_class_weights(y, alpha)

        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            fold_models = []
            for k in range(1, self.num_classes):
                fold_models.append(
                    self._fit_one_threshold(
                        X_tr, y_tr, X_va, y_va, k=k, class_w=class_w, pos_mult=pos_mult
                    )
                )

            cum_probs = self._predict_cum_probs_with_models(X_va, fold_models)
            y_pred = _decode_coral(cum_probs, tau=float(tau))
            scores.append(_qwk(y_va, y_pred))

        return float(np.mean(scores))

    def tune_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        best = {"alpha": None, "pos_mult": None, "tau": None, "score_qwk": -np.inf}
        for alpha in self.cfg.alpha_grid:
            for pos_mult in self.cfg.pos_mult_grid:
                for tau in self.cfg.tau_grid:
                    score = self._cv_score_params(X, y, alpha=float(alpha), pos_mult=float(pos_mult), tau=float(tau))
                    if score > best["score_qwk"]:
                        best = {"alpha": float(alpha), "pos_mult": float(pos_mult), "tau": float(tau), "score_qwk": float(score)}
        return best

    def fit(self, X, y, tune_weights: bool = True) -> "LGBMCoralModel":
        Xn = _to_numpy(X)
        yn = _ensure_int_labels(y)

        K = self.num_classes
        if yn.min() < 0 or yn.max() > (K - 1):
            raise ValueError(f"y must be in [0, {K-1}] for num_classes={K}. Got min={yn.min()}, max={yn.max()}.")

        has_enough_per_class = _has_min_count_per_class(yn, self.cfg.cv_folds)

        do_weight_tune = (
            bool(tune_weights)
            and has_enough_per_class
            and (len(yn) >= self.cfg.min_rows_per_fold)
            and (self.cfg.cv_folds >= 3)
        )

        if do_weight_tune:
            best = self.tune_weights(Xn, yn)
            best["tuning_skipped"] = False
        else:
            best = {
                "alpha": 1.0,
                "pos_mult": 1.0,
                "tau": 0.5,          # may be overwritten by decode-only tuning
                "score_qwk": None,
                "tuning_skipped": True,
                "reason": "min_count_per_class < cv_folds" if not has_enough_per_class else "insufficient_rows_or_folds",
            }

        # Create a stratified holdout split for early stopping + optional decode-only tau tuning
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.cfg.random_state)
        tr_idx, va_idx = next(skf.split(Xn, yn))
        X_tr, X_va = Xn[tr_idx], Xn[va_idx]
        y_tr, y_va = yn[tr_idx], yn[va_idx]

        class_w = _build_class_weights(yn, best["alpha"])

        # Fit final threshold models using early stopping on holdout
        self.models_ = []
        for k in range(1, self.num_classes):
            self.models_.append(
                self._fit_one_threshold(
                    X_tr, y_tr, X_va, y_va, k=k, class_w=class_w, pos_mult=best["pos_mult"]
                )
            )

        # Decode-only tuning (tau) if weight tuning was skipped
        if best.get("tuning_skipped", False):
            if np.unique(y_va).size >= 2:
                cum_va = self._predict_cum_probs_with_models(X_va, self.models_)
                tau_info = self._tune_tau_on_valid(y_valid=y_va, cum_probs_valid=cum_va)
                best["tau"] = tau_info["tau"]
                best["tau_score_qwk_valid"] = tau_info["tau_score_qwk"]
                best["decode_only_tuned"] = True
            else:
                best["decode_only_tuned"] = False
                best["tau_score_qwk_valid"] = None

        self.best_params_ = best
        self.fitted_ = True
        return self

    def predict_cumproba(self, X) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted.")
        Xn = _to_numpy(X)
        return self._predict_cum_probs_with_models(Xn, self.models_)

    def predict(self, X, tau: Optional[float] = None) -> np.ndarray:
        if tau is None:
            tau = float(self.best_params_.get("tau", 0.5))
        cum_probs = self.predict_cumproba(X)
        return _decode_coral(cum_probs, tau=float(tau))

    def predict_proba(self, X) -> np.ndarray:
        """
        Convert cumulative probs to class probs:
          p0 = 1 - P(y>=1)
          pk = P(y>=k) - P(y>=k+1) for k=1..K-2
          pK-1 = P(y>=K-1)
        """
        cum = self.predict_cumproba(X)  # (n, K-1)
        n = cum.shape[0]
        K = self.num_classes
        proba = np.zeros((n, K), dtype=float)

        proba[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, K - 1):
            proba[:, k] = cum[:, k - 1] - cum[:, k]
        proba[:, K - 1] = cum[:, K - 2]

        proba = np.clip(proba, 0.0, 1.0)
        row_sum = proba.sum(axis=1, keepdims=True)
        proba = proba / np.clip(row_sum, 1e-12, None)
        return proba

# train_predict_with_pca_gate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lightgbm import LGBMClassifier

from lgbm_coral import LGBMCoralModel, LGBMCoralConfig


# ----------------------------
# Load config.py dynamically
# ----------------------------
import importlib.util

def load_config(path: str = "config.py"):
    spec = importlib.util.spec_from_file_location("config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg = load_config("config.py")

TRAIN_CSV = cfg.TRAIN_CSV
UNSEEN_CSV = getattr(cfg, "UNSEEN_CSV", None)

TARGETS_CFG = getattr(cfg, "TARGETS", None)
TARGET_PREFIX = getattr(cfg, "TARGET_PREFIX", "y_")
ID_COL = getattr(cfg, "ID_COL", None)
ID_PREFIX = getattr(cfg, "ID_PREFIX", "id_")
FEATURE_PREFIX = getattr(cfg, "FEATURE_PREFIX", "ft_")

ARTIFACTS_DIR = getattr(cfg, "ARTIFACTS_DIR", "artifacts")
MODELS_DIR = getattr(cfg, "MODELS_DIR", f"{ARTIFACTS_DIR}/coral_models")
OUT_DIR = getattr(cfg, "OUT_DIR", f"{ARTIFACTS_DIR}/final_predictions")
DECODE_GRID = getattr(cfg, "DECODE_GRID", None)

STAGE0_PATH = getattr(cfg, "STAGE0_PATH", None)
FEATURE_COLS_FROM_CONFIG = getattr(cfg, "FEATURE_COLS", None)

MIN_ROWS_PER_TARGET = int(getattr(cfg, "MIN_ROWS_PER_TARGET", 80))

# PCA gate knobs
PCA_GATE_DIR = getattr(cfg, "PCA_GATE_DIR", f"{ARTIFACTS_DIR}/pca_gate")
PCA_N_COMPONENTS = int(getattr(cfg, "PCA_N_COMPONENTS", 20))
PCA_TOP_K_LOADINGS = int(getattr(cfg, "PCA_TOP_K_LOADINGS", 15))
PCA_PLOT_MAX = int(getattr(cfg, "PCA_PLOT_MAX", 50))

DROP_MISSING_FRAC = float(getattr(cfg, "PCA_DROP_MISSING_FRAC", 0.80))
DROP_DOMINANT_PC_COUNT = int(getattr(cfg, "PCA_DROP_DOMINANT_PC_COUNT", 3))
DROP_LOW_VAR_QUANTILE = float(getattr(cfg, "PCA_DROP_LOW_VAR_QUANTILE", 0.00))

PCA_ALWAYS_KEEP = set(getattr(cfg, "PCA_ALWAYS_KEEP", []))

# Safety check knobs
SAFETY_CHECK_ENABLED = bool(getattr(cfg, "PCA_SAFETY_CHECK_ENABLED", True))
SAFETY_TOP_N = int(getattr(cfg, "PCA_SAFETY_TOP_N", 50))
SAFETY_MAX_TARGETS = int(getattr(cfg, "PCA_SAFETY_MAX_TARGETS", 50))
SAFETY_MIN_ROWS = int(getattr(cfg, "PCA_SAFETY_MIN_ROWS", 300))
SAFETY_SAMPLE_ROWS = int(getattr(cfg, "PCA_SAFETY_SAMPLE_ROWS", 8000))
SAFETY_N_ESTIMATORS = int(getattr(cfg, "PCA_SAFETY_N_ESTIMATORS", 400))
SAFETY_RANDOM_STATE = int(getattr(cfg, "PCA_SAFETY_RANDOM_STATE", 42))

SAFETY_MIN_TARGET_SUPPORT = int(getattr(cfg, "PCA_SAFETY_MIN_TARGET_SUPPORT", 0))
SAFETY_WEIGHTED_SUPPORT_ENABLED = bool(getattr(cfg, "PCA_SAFETY_WEIGHTED_SUPPORT_ENABLED", True))
SAFETY_WEIGHTED_SUPPORT_THRESHOLD = float(getattr(cfg, "PCA_SAFETY_WEIGHTED_SUPPORT_THRESHOLD", 0))

# Prediction output controls
OUTPUT_PROB_COLUMNS = bool(getattr(cfg, "OUTPUT_PROB_COLUMNS", True))
MAX_PROB_CLASSES = int(getattr(cfg, "MAX_PROB_CLASSES", 10))


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)

def align_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in feature_cols if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    return out[feature_cols]

def detect_id_col(df: pd.DataFrame) -> Optional[str]:
    if ID_COL is not None:
        return ID_COL if ID_COL in df.columns else None
    # auto-detect first id_ column
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    return id_cols[0] if id_cols else None

def detect_targets(df: pd.DataFrame) -> List[str]:
    if TARGETS_CFG is not None:
        return list(TARGETS_CFG)
    return [c for c in df.columns if c.startswith(TARGET_PREFIX)]

def get_feature_cols(df: pd.DataFrame, targets: List[str], id_col: Optional[str]) -> List[str]:
    # Priority 1: explicit list
    if FEATURE_COLS_FROM_CONFIG is not None:
        return list(FEATURE_COLS_FROM_CONFIG)

    # Priority 2: stage0 artifact
    if STAGE0_PATH is not None and Path(STAGE0_PATH).exists():
        obj = joblib.load(STAGE0_PATH)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ["kept_features", "features", "selected_features"]:
                if k in obj and isinstance(obj[k], list):
                    return obj[k]

    # Priority 3: prefix-based features
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if ft_cols:
        return ft_cols

    # Fallback: exclude targets and id
    return [c for c in df.columns if c not in targets and c != id_col]


# ----------------------------
# Ordinal label remap per target
# ----------------------------
def make_ordinal_codes(y: pd.Series) -> Tuple[np.ndarray, Dict]:
    y_nonnull = y.dropna()
    classes = sorted(y_nonnull.unique().tolist())
    to_code = {c: i for i, c in enumerate(classes)}
    from_code = {i: c for c, i in to_code.items()}
    y_code = y.map(to_code).astype(int).to_numpy()
    meta = {
        "classes_sorted": classes,
        "to_code": {str(k): int(v) for k, v in to_code.items()},
        "from_code": {str(k): v for k, v in from_code.items()},
        "n_classes": len(classes),
        "class_counts": y_nonnull.value_counts().to_dict(),
    }
    return y_code, meta

def decode_to_original_labels(pred_code: np.ndarray, from_code: Dict[str, object]) -> np.ndarray:
    mapping = {int(k): v for k, v in from_code.items()}
    return np.array([mapping.get(int(i), np.nan) for i in pred_code], dtype=object)


# ----------------------------
# Safety check: protect predictive features
# ----------------------------
def safety_check_protect_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    targets: List[str],
) -> Dict[str, object]:
    if not SAFETY_CHECK_ENABLED:
        return {"protected_features": set(), "per_target_top": {}, "used_targets": []}

    rng = np.random.RandomState(SAFETY_RANDOM_STATE)

    # choose targets with enough labels, prioritize by labeled count
    t_counts = []
    for t in targets:
        if t in train_df.columns:
            n = int(train_df[t].notna().sum())
            if n >= SAFETY_MIN_ROWS:
                t_counts.append((t, n))
    t_counts.sort(key=lambda x: x[1], reverse=True)
    chosen = [t for t, _ in t_counts[:SAFETY_MAX_TARGETS]]

    base_params = dict(
        n_estimators=SAFETY_N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SAFETY_RANDOM_STATE,
    )

    protected: set[str] = set()
    per_target_top: Dict[str, List[str]] = {}
    used_targets: List[str] = []

    for t in chosen:
        mask = train_df[t].notna()
        y_raw = train_df.loc[mask, t]
        y_code, meta = make_ordinal_codes(y_raw)
        K = int(meta["n_classes"])
        if K < 2:
            continue

        X = align_features(train_df.loc[mask], feature_cols)
        X = coerce_numeric(X)

        n_rows = X.shape[0]
        if n_rows > SAFETY_SAMPLE_ROWS:
            idx = rng.choice(np.arange(n_rows), size=SAFETY_SAMPLE_ROWS, replace=False)
            X = X.iloc[idx]
            y_code = y_code[idx]

        # median impute
        X_imp = X.fillna(X.median(numeric_only=True))

        # mild inverse frequency weights
        vals, counts = np.unique(y_code, return_counts=True)
        freq = counts / counts.sum()
        inv = 1.0 / np.clip(freq, 1e-12, None)
        inv = inv / np.mean(inv)
        cw = {int(v): float(w) for v, w in zip(vals, inv)}
        sw = np.array([cw[int(v)] for v in y_code], dtype=float)

        clf = LGBMClassifier(objective="multiclass", num_class=K, **base_params)
        clf.fit(X_imp, y_code, sample_weight=sw)

        gain = clf.booster_.feature_importance(importance_type="gain")
        imp = pd.Series(gain, index=feature_cols).sort_values(ascending=False)

        top_feats = imp.head(SAFETY_TOP_N).index.tolist()
        per_target_top[t] = top_feats
        protected.update(top_feats)
        used_targets.append(t)

    return {"protected_features": protected, "per_target_top": per_target_top, "used_targets": used_targets}


# ----------------------------
# PCA gate with safety + support + weighted support
# ----------------------------
def pca_gate_build_clean_features(
    train_X: pd.DataFrame,
    unseen_X: Optional[pd.DataFrame],
    feature_cols: List[str],
    pca_out_dir: Path,
    train_df_raw: pd.DataFrame,
    targets: List[str],
) -> Dict[str, object]:
    pca_out_dir.mkdir(parents=True, exist_ok=True)

    # ---- basic stats on train
    X_train_num = coerce_numeric(train_X)
    feat_missing = X_train_num.isna().mean()
    feat_var = X_train_num.var(axis=0, skipna=True).fillna(0.0)

    feat_missing.to_csv(pca_out_dir / "feature_missingness.csv")
    feat_var.to_csv(pca_out_dir / "feature_variance.csv")

    high_missing = set(feat_missing[feat_missing >= DROP_MISSING_FRAC].index.tolist())

    low_var = set()
    if DROP_LOW_VAR_QUANTILE and DROP_LOW_VAR_QUANTILE > 0:
        q = float(np.quantile(feat_var.values, DROP_LOW_VAR_QUANTILE))
        low_var = set(feat_var[feat_var <= q].index.tolist())

    # ---- fit PCA on train
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_imp = imputer.fit_transform(X_train_num.values)
    X_scaled = scaler.fit_transform(X_imp)

    n_comp = min(PCA_N_COMPONENTS, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    Z_train = pca.fit_transform(X_scaled)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pd.DataFrame({"pc": np.arange(1, len(evr) + 1), "evr": evr, "cum_evr": cum}).to_csv(
        pca_out_dir / "explained_variance.csv", index=False
    )
    m = min(PCA_PLOT_MAX, len(cum))
    plt.figure()
    plt.plot(np.arange(1, m + 1), cum[:m])
    plt.xlabel("PC")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance (fit on train)")
    plt.tight_layout()
    plt.savefig(pca_out_dir / "explained_variance.png", dpi=160)
    plt.close()

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    loadings.to_csv(pca_out_dir / "pca_loadings_matrix.csv")

    rows = []
    for pc in loadings.columns:
        top_pos = loadings[pc].nlargest(PCA_TOP_K_LOADINGS)
        top_neg = loadings[pc].nsmallest(PCA_TOP_K_LOADINGS)
        for f, v in top_pos.items():
            rows.append({"pc": pc, "sign": "+", "feature": f, "loading": float(v)})
        for f, v in top_neg.items():
            rows.append({"pc": pc, "sign": "-", "feature": f, "loading": float(v)})
    pd.DataFrame(rows).to_csv(pca_out_dir / "pca_top_loadings.csv", index=False)

    abs_load = loadings.abs()
    top3_mask = (abs_load.rank(axis=0, ascending=False) <= 3)
    dominant_count = top3_mask.sum(axis=1)
    dominant = set(dominant_count[dominant_count >= DROP_DOMINANT_PC_COUNT].index.tolist())

    # missingness scatter
    row_missing = train_X.isna().mean(axis=1).values
    plt.figure()
    sc = plt.scatter(Z_train[:, 0], Z_train[:, 1], c=row_missing, s=10, alpha=0.85)
    plt.colorbar(sc, label="Row missingness ratio")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Train PCA (PC1 vs PC2) colored by missingness")
    plt.tight_layout()
    plt.savefig(pca_out_dir / "train_pc1_pc2_missingness.png", dpi=160)
    plt.close()

    # drift plot
    drift = {}
    if unseen_X is not None and len(unseen_X) > 0:
        X_unseen_num = coerce_numeric(unseen_X)
        Xu_imp = imputer.transform(X_unseen_num.values)
        Xu_scaled = scaler.transform(Xu_imp)
        Z_unseen = pca.transform(Xu_scaled)

        plt.figure()
        plt.scatter(Z_train[:, 0], Z_train[:, 1], s=10, alpha=0.45, label="train")
        plt.scatter(Z_unseen[:, 0], Z_unseen[:, 1], s=10, alpha=0.45, label="unseen")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA drift: Train vs Unseen (PCA fit on train)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pca_out_dir / "drift_train_vs_unseen_pc1_pc2.png", dpi=160)
        plt.close()

        n_drift_pcs = min(10, Z_train.shape[1])
        mu_tr = Z_train[:, :n_drift_pcs].mean(axis=0)
        mu_un = Z_unseen[:, :n_drift_pcs].mean(axis=0)
        sd_tr = Z_train[:, :n_drift_pcs].std(axis=0, ddof=0)
        sd_tr = np.where(sd_tr == 0, 1.0, sd_tr)
        z_mean_shift = (mu_un - mu_tr) / sd_tr
        pd.DataFrame({
            "pc": np.arange(1, n_drift_pcs + 1),
            "mean_train": mu_tr,
            "mean_unseen": mu_un,
            "std_train": sd_tr,
            "z_mean_shift": z_mean_shift,
        }).to_csv(pca_out_dir / "drift_summary_top_pcs.csv", index=False)

        drift = {"top_pcs": int(n_drift_pcs), "mean_shift_l2": float(np.linalg.norm(z_mean_shift))}

    # ---- candidates before safety
    candidates = (high_missing | dominant | low_var) - set(PCA_ALWAYS_KEEP)

    # ---- safety protections
    safety = safety_check_protect_features(train_df=train_df_raw, feature_cols=feature_cols, targets=targets)
    protected_by_importance = set(safety.get("protected_features", set()))
    per_target_top = safety.get("per_target_top", {})
    used_targets = safety.get("used_targets", [])

    if SAFETY_CHECK_ENABLED:
        with open(pca_out_dir / "safety_check_top_features_per_target.json", "w", encoding="utf-8") as f:
            json.dump(per_target_top, f, indent=2)

    # support counts
    support_counts: Dict[str, int] = {}
    for _t, feats in per_target_top.items():
        for f in feats:
            support_counts[f] = support_counts.get(f, 0) + 1

    support_counts_path = pca_out_dir / "feature_support_counts.csv"
    if support_counts:
        pd.Series(support_counts).sort_values(ascending=False).to_csv(
            support_counts_path, header=["target_support_count"]
        )
    else:
        pd.Series(dtype=int).to_csv(support_counts_path, header=["target_support_count"])

    globally_supported_by_count = set()
    if SAFETY_CHECK_ENABLED and SAFETY_MIN_TARGET_SUPPORT and SAFETY_MIN_TARGET_SUPPORT > 0:
        globally_supported_by_count = {f for f, c in support_counts.items() if c >= SAFETY_MIN_TARGET_SUPPORT}

    # weighted support
    weighted_support: Dict[str, float] = {}
    globally_supported_weighted = set()
    if SAFETY_CHECK_ENABLED and SAFETY_WEIGHTED_SUPPORT_ENABLED and SAFETY_WEIGHTED_SUPPORT_THRESHOLD > 0:
        for _t, feats in per_target_top.items():
            for rank, f in enumerate(feats):
                add = max(SAFETY_TOP_N - rank, 0)
                weighted_support[f] = weighted_support.get(f, 0.0) + float(add)
        globally_supported_weighted = {f for f, s in weighted_support.items() if s >= SAFETY_WEIGHTED_SUPPORT_THRESHOLD}

    weighted_path = pca_out_dir / "feature_weighted_support.csv"
    if weighted_support:
        pd.Series(weighted_support).sort_values(ascending=False).to_csv(
            weighted_path, header=["weighted_support_score"]
        )
    else:
        pd.Series(dtype=float).to_csv(weighted_path, header=["weighted_support_score"])

    # final never-drop set
    never_drop = set(PCA_ALWAYS_KEEP) | protected_by_importance | globally_supported_by_count | globally_supported_weighted

    final_drop = set(candidates) - never_drop
    cleaned_feature_cols = [c for c in feature_cols if c not in final_drop]

    # write lists
    with open(pca_out_dir / "feature_cols_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_feature_cols, f, indent=2)

    pd.Series(sorted(list(candidates))).to_csv(
        pca_out_dir / "drop_candidates_before_safety.csv", index=False, header=["feature"]
    )
    pd.Series(sorted(list(final_drop))).to_csv(
        pca_out_dir / "dropped_features_final.csv", index=False, header=["feature"]
    )

    report = {
        "n_features_in": int(len(feature_cols)),
        "n_features_cleaned": int(len(cleaned_feature_cols)),
        "drop_candidates_count": int(len(candidates)),
        "dropped_count_final": int(len(final_drop)),
        "rules": {
            "drop_missing_frac": DROP_MISSING_FRAC,
            "drop_dominant_pc_count": DROP_DOMINANT_PC_COUNT,
            "drop_low_var_quantile": DROP_LOW_VAR_QUANTILE,
        },
        "sets": {
            "high_missing": sorted(list(high_missing)),
            "dominant_in_pcs": sorted(list(dominant)),
            "low_variance": sorted(list(low_var)),
            "drop_candidates_before_safety": sorted(list(candidates)),
            "protected_by_importance": sorted(list(protected_by_importance)),
            "globally_supported_by_count": sorted(list(globally_supported_by_count)),
            "globally_supported_weighted": sorted(list(globally_supported_weighted)),
            "always_keep": sorted(list(PCA_ALWAYS_KEEP)),
            "final_dropped": sorted(list(final_drop)),
        },
        "safety_check": {
            "enabled": bool(SAFETY_CHECK_ENABLED),
            "top_n": int(SAFETY_TOP_N),
            "used_targets": used_targets,
            "min_target_support": int(SAFETY_MIN_TARGET_SUPPORT),
            "weighted_support_enabled": bool(SAFETY_WEIGHTED_SUPPORT_ENABLED),
            "weighted_support_threshold": float(SAFETY_WEIGHTED_SUPPORT_THRESHOLD),
            "globally_supported_by_count_count": int(len(globally_supported_by_count)),
            "globally_supported_weighted_count": int(len(globally_supported_weighted)),
        },
        "drift": drift,
        "paths": {
            "feature_cols_cleaned": str(pca_out_dir / "feature_cols_cleaned.json"),
            "pca_dir": str(pca_out_dir),
            "feature_support_counts": str(support_counts_path),
            "feature_weighted_support": str(weighted_path),
        },
    }
    with open(pca_out_dir / "pca_gate_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {"cleaned_feature_cols": cleaned_feature_cols, "final_drop": final_drop, "report": report}


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(MODELS_DIR)
    ensure_dir(OUT_DIR)
    pca_out = ensure_dir(PCA_GATE_DIR)

    train_df = pd.read_csv(TRAIN_CSV)
    unseen_df = safe_read_csv(UNSEEN_CSV)

    id_col = detect_id_col(train_df)
    targets = detect_targets(train_df)

    feature_cols = get_feature_cols(train_df, targets=targets, id_col=id_col)

    # align for PCA gate
    X_train_all = align_features(train_df, feature_cols)
    X_unseen_all = align_features(unseen_df, feature_cols) if unseen_df is not None else None

    gate = pca_gate_build_clean_features(
        train_X=X_train_all,
        unseen_X=X_unseen_all,
        feature_cols=feature_cols,
        pca_out_dir=pca_out,
        train_df_raw=train_df,
        targets=targets,
    )
    feature_cols_cleaned = gate["cleaned_feature_cols"]

    # build full df
    if unseen_df is not None:
        full_df = pd.concat([train_df, unseen_df], ignore_index=True)
        full_source = np.array(["train"] * len(train_df) + ["unseen"] * len(unseen_df), dtype=object)
    else:
        full_df = train_df.copy()
        full_source = np.array(["train"] * len(train_df), dtype=object)

    X_full = align_features(full_df, feature_cols_cleaned)

    out_df = full_df.copy()
    out_df["_source"] = full_source

    # Save final feature cols
    with open(Path(OUT_DIR) / "feature_cols_used_final.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols_cleaned, f, indent=2)

    coral_cfg = LGBMCoralConfig()
    decode_grid = list(DECODE_GRID) if DECODE_GRID is not None else None

    summary_rows = []

    # loop each target
    for t in targets:
        if t not in train_df.columns:
            continue

        mask = train_df[t].notna()
        n_rows = int(mask.sum())
        if n_rows < MIN_ROWS_PER_TARGET:
            summary_rows.append({"target": t, "status": "skip_too_few_rows", "n_rows": n_rows})
            continue

        X_train_t = align_features(train_df.loc[mask], feature_cols_cleaned)
        y_train_t = train_df.loc[mask, t].copy()

        y_code, label_meta = make_ordinal_codes(y_train_t)
        K = int(label_meta["n_classes"])
        if K < 2:
            summary_rows.append({"target": t, "status": "skip_single_class", "n_rows": n_rows, "n_classes": K})
            continue

        try:
            model = LGBMCoralModel(num_classes=K, config=coral_cfg, decode_grid=decode_grid)
            model.fit(X_train_t, y_code, tune_weights=True)
        except Exception as e:
            summary_rows.append({"target": t, "status": "error_fit", "n_rows": n_rows, "n_classes": K, "error": repr(e)})
            continue

        pred_code = model.predict(X_full)
        pred_label = decode_to_original_labels(pred_code, label_meta["from_code"])

        out_df[f"{t}_pred"] = pred_label
        out_df[f"{t}_pred_code"] = pred_code
        out_df[f"{t}_tau_used"] = float(model.best_params_.get("tau", 0.5))

        if OUTPUT_PROB_COLUMNS and K <= MAX_PROB_CLASSES:
            cum = model.predict_cumproba(X_full)
            proba = model.predict_proba(X_full)
            for k in range(1, K):
                out_df[f"{t}_p_ge_{k}"] = cum[:, k - 1].astype(float)
            for k in range(K):
                out_df[f"{t}_p_class_{k}"] = proba[:, k].astype(float)

        model_path = Path(MODELS_DIR) / f"{t}_coral.joblib"
        meta_path = Path(MODELS_DIR) / f"{t}_meta.json"
        joblib.dump(model, model_path)

        meta_out = {
            "target": t,
            "n_rows": n_rows,
            "n_features": int(X_train_t.shape[1]),
            "label_meta": label_meta,
            "best_params": model.best_params_,
            "feature_cols_cleaned_path": str(Path(PCA_GATE_DIR) / "feature_cols_cleaned.json"),
            "feature_cols_cleaned_count": len(feature_cols_cleaned),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        if id_col and id_col in train_df.columns:
            train_df.loc[mask, [id_col, t]].to_csv(Path(MODELS_DIR) / f"{t}_train_rows.csv", index=False)

        summary_rows.append({
            "target": t,
            "status": "trained_and_predicted",
            "n_rows": n_rows,
            "n_classes": K,
            "tau": float(model.best_params_.get("tau", 0.5)),
            "tuning_skipped": bool(model.best_params_.get("tuning_skipped", False)),
            "decode_only_tuned": bool(model.best_params_.get("decode_only_tuned", False)),
            "prob_cols_written": bool(OUTPUT_PROB_COLUMNS and K <= MAX_PROB_CLASSES),
        })

    preds_path = Path(OUT_DIR) / "full_predictions.csv"
    out_df.to_csv(preds_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(OUT_DIR) / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Targets detected:         {len(targets)}")
    print(f"[OK] Features (raw->clean):   {len(feature_cols)} -> {len(feature_cols_cleaned)}")
    print(f"[OK] PCA gate outputs:        {Path(PCA_GATE_DIR).resolve()}")
    print(f"[OK] Saved predictions:       {preds_path.resolve()}")
    print(f"[OK] Saved summary:           {summary_path.resolve()}")
    print(f"[OK] Saved models to:         {Path(MODELS_DIR).resolve()}")


if __name__ == "__main__":
    main()




