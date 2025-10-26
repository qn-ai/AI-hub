# erccfs.py
from typing import Callable, Optional, Dict, Any, List
import numpy as np

Array = np.ndarray


class ERCCFS:
    def __init__(
        self,
        compute_A: Callable[[Array, Array, Array, Array, float, Dict[str, Any]], Array],
        compute_B: Callable[[Array, Array, Array, Array, float, Dict[str, Any]], Array],
        update_W: Callable[[Array, Array, Array, Array, Array, float, float, float, Dict[str, Any]], Array],
        compute_P: Optional[Callable[[Array, Array, Array, Dict[str, Any]], Array]] = None,
        n_features: Optional[int] = None,
        feature_ratio: Optional[float] = None,
        stability_overlap: float = 0.9,
        max_outer_iters: int = 50,
        max_inner_iters: int = 200,
        tol_inner: float = 1e-6,
        rho1_init: float = 1e-6,
        rho2_init: float = 1e-6,
        rho_max: float = 1e10,
        alpha: float = 1e2,
        tau: float = 1.1,
        ridge_lambda: float = 1e-6,
    ):
        self.compute_A = compute_A
        self.compute_B = compute_B
        self.update_W = update_W
        self.compute_P = compute_P
        self.n_features = n_features
        self.feature_ratio = feature_ratio
        self.stability_overlap = stability_overlap
        self.max_outer_iters = max_outer_iters
        self.max_inner_iters = max_inner_iters
        self.tol_inner = tol_inner
        self.rho1_init = rho1_init
        self.rho2_init = rho2_init
        self.rho_max = rho_max
        self.alpha = alpha
        self.tau = tau
        self.ridge_lambda = ridge_lambda

        self.W_: Optional[Array] = None
        self.A_: Optional[Array] = None
        self.B_: Optional[Array] = None
        self.Lambda1_: Optional[Array] = None
        self.Lambda2_: Optional[Array] = None
        self.P_: Optional[Array] = None
        self.selected_features_: Optional[np.ndarray] = None
        self.history_: List[Dict[str, Any]] = []

    def _init_least_squares(self, X: Array, Y: Array) -> Array:
        d = X.shape[1]
        XtX = X.T @ X
        reg = self.ridge_lambda * np.eye(d)
        XtY = X.T @ Y
        return np.linalg.solve(XtX + reg, XtY)

    def _select_features_by_row_norm(self, W: Array, k: int) -> np.ndarray:
        norms = np.linalg.norm(W, axis=1)
        idx = np.argsort(-norms)[:k]
        return np.sort(idx)

    def fit(self, X: Array, Y: Array, *, user_ctx: Optional[Dict[str, Any]] = None):
        if user_ctx is None:
            user_ctx = {}
        n, d = X.shape
        m = Y.shape[1]

        W = self._init_least_squares(X, Y)
        B = W.copy()
        A = W.T.copy()
        Lambda1 = np.zeros((m, d))
        Lambda2 = np.zeros((d, m))
        rho1, rho2 = self.rho1_init, self.rho2_init

        if self.compute_P is None:
            class_means = []
            for j in range(m):
                mask = Y[:, j] > 0.5
                class_means.append(X[mask].mean(axis=0) if np.any(mask) else np.zeros(d))
            class_means = np.stack(class_means, axis=0)
            P = np.corrcoef(class_means)
        else:
            P = self.compute_P(X, Y, W, user_ctx)

        prev_selected, selected = None, None

        for outer in range(self.max_outer_iters):
            for inner in range(self.max_inner_iters):
                A_new = self.compute_A(W, A, Lambda1, P, rho1, user_ctx)  # Eq. 15
                B_new = self.compute_B(W, B, Lambda2, P, rho2, user_ctx)  # Eq. 17
                W_new = self.update_W(X, Y, A_new, B_new, P, rho1, rho2, self.alpha, user_ctx)  # Eq. 19

                Lambda1 = Lambda1 + rho1 * (W_new.T - A_new)
                Lambda2 = Lambda2 + rho2 * (W_new - B_new)

                delta = max(
                    np.linalg.norm(A_new - A, ord="fro"),
                    np.linalg.norm(B_new - B, ord="fro"),
                    np.linalg.norm(W_new - W, ord="fro"),
                )
                A, B, W = A_new, B_new, W_new

                rho1 = min(self.tau * rho1, self.rho_max)
                rho2 = min(self.tau * rho2, self.rho_max)

                self.history_.append(
                    {"outer": outer, "inner": inner, "delta": float(delta), "rho1": float(rho1), "rho2": float(rho2)}
                )
                if delta < self.tol_inner:
                    break

            k = self.n_features if self.n_features is not None else max(1, int((self.feature_ratio or 0.2) * d))
            selected = self._select_features_by_row_norm(W, k)

            if self.compute_P is None:
                Xr = X[:, selected]
                class_means_r = []
                for j in range(m):
                    mask = Y[:, j] > 0.5
                    class_means_r.append(Xr[mask].mean(axis=0) if np.any(mask) else np.zeros(Xr.shape[1]))
                class_means_r = np.stack(class_means_r, axis=0)
                P = np.corrcoef(class_means_r + 1e-12)
            else:
                P = self.compute_P(X[:, selected], Y, W, user_ctx)

            if prev_selected is not None:
                overlap = len(np.intersect1d(prev_selected, selected)) / float(len(np.union1d(prev_selected, selected)))
                if overlap >= self.stability_overlap:
                    prev_selected = selected
                    break
            prev_selected = selected

        self.W_ = W
        self.A_ = A
        self.B_ = B
        self.Lambda1_ = Lambda1
        self.Lambda2_ = Lambda2
        self.P_ = P
        self.selected_features_ = prev_selected if prev_selected is not None else selected
        return self


# -------- placeholders: replace with the paper’s Eq. (15), (17), (19) --------
def placeholder_compute_A(W: Array, A: Array, Lambda1: Array, P: Array, rho1: float, ctx: Dict[str, Any]) -> Array:
    return W.T + (1.0 / max(rho1, 1e-12)) * Lambda1

def placeholder_compute_B(W: Array, B: Array, Lambda2: Array, P: Array, rho2: float, ctx: Dict[str, Any]) -> Array:
    return W + (1.0 / max(rho2, 1e-12)) * Lambda2

def placeholder_update_W(X: Array, Y: Array, A: Array, B: Array, P: Array,
                         rho1: float, rho2: float, alpha: float, ctx: Dict[str, Any]) -> Array:
    XtX = X.T @ X
    M = XtX + (rho1 + rho2 + alpha) * np.eye(X.shape[1])
    RHS = X.T @ Y + rho1 * A.T + rho2 * B
    return np.linalg.solve(M, RHS)

def make_placeholder_erccfs(**kwargs) -> ERCCFS:
    return ERCCFS(
        compute_A=placeholder_compute_A,
        compute_B=placeholder_compute_B,
        update_W=placeholder_update_W,
        **kwargs
    )




# select_features_erccfs.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from erccfs import make_placeholder_erccfs

def parse_list(arg: str):
    return [x.strip() for x in arg.split(",") if x.strip()]

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise ValueError("Non-numeric values found after coercion; please clean your CSV.")
    return df

def main():
    ap = argparse.ArgumentParser(description="ERCCFS feature selection w/ one-hot or single-label input")
    ap.add_argument("--csv", required=True, help="CSV path")
    # feature columns
    ap.add_argument("--feature-cols", help="Comma-separated names of feature columns (46 expected)")
    ap.add_argument("--feature-prefix", help="Prefix to select feature columns (e.g., f)")
    # labels: either single or multi
    ap.add_argument("--label-col", help="Single label column (integer classes)")
    ap.add_argument("--label-cols", help="Comma-separated names of one-hot label columns (57 expected)")
    ap.add_argument("--label-prefix", help="Prefix for one-hot label columns (e.g., y)")
    # algo/vis
    ap.add_argument("--k", type=int, default=15, help="Top-k features to select")
    ap.add_argument("--outdir", default="erccfs_out", help="Output directory")
    ap.add_argument("--max-outer", type=int, default=10)
    ap.add_argument("--max-inner", type=int, default=100)
    ap.add_argument("--tol-inner", type=float, default=1e-5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # --- pick feature columns ---
    if args.feature_cols:
        feature_cols = parse_list(args.feature_cols)
    elif args.feature_prefix:
        feature_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    else:
        # fallback: assume all but provided label columns
        if args.label_cols:
            label_cols = set(parse_list(args.label_cols))
        elif args.label_prefix:
            label_cols = set([c for c in df.columns if c.startswith(args.label_prefix)])
        elif args.label_col:
            label_cols = {args.label_col}
        else:
            raise ValueError("Specify --feature-cols or --feature-prefix (or provide label args so features can be inferred).")
        feature_cols = [c for c in df.columns if c not in label_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns selected.")
    if len(feature_cols) != 46:
        print(f"[warn] Expected 46 features, got {len(feature_cols)}. Proceeding.")

    X_df = ensure_numeric(df[feature_cols])

    # --- pick labels (single col or one-hot) ---
    if args.label_cols or args.label_prefix:
        # one-hot multi-class
        if args.label_cols:
            label_cols = parse_list(args.label_cols)
        else:
            label_cols = [c for c in df.columns if c.startswith(args.label_prefix)]
        if len(label_cols) == 0:
            raise ValueError("No label columns found for one-hot labels.")
        Y_df = ensure_numeric(df[label_cols])
        # Ensure binary-ish (0/1)
        Y = Y_df.to_numpy(dtype=float)
        m = Y.shape[1]
        if m != 57:
            print(f"[warn] One-hot label width is {m}, not 57. Proceeding.")
        # Optional sanity: row sums should be ~1
        rs = Y.sum(axis=1)
        if not np.all((np.abs(rs - 1) < 1e-6) | (rs == 0)):
            print("[warn] One-hot rows do not sum to 1 exactly. Make sure labels are proper one-hot.")
    elif args.label_col:
        # single label -> one-hot
        y = df[args.label_col].to_numpy()
        classes = np.unique(y)
        m = len(classes)
        mapping = {c: i for i, c in enumerate(classes)}
        y_idx = np.vectorize(mapping.get)(y)
        Y = np.eye(m)[y_idx]
        if m != 57:
            print(f"[warn] Found {m} classes; expected 57. Proceeding.")
    else:
        raise ValueError("Provide either --label-cols/--label-prefix (one-hot) or --label-col (single).")

    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)

    # --- run ERCCFS ---
    model = make_placeholder_erccfs(
        n_features=args.k,
        stability_overlap=0.85,
        max_outer_iters=args.max_outer,
        max_inner_iters=args.max_inner,
        tol_inner=args.tol_inner
    )
    model.fit(X, Y)

    selected_idx = model.selected_features_
    W = model.W_
    row_norms = np.linalg.norm(W, axis=1)

    selected_features = [feature_names[i] for i in selected_idx]
    selected_scores = [float(row_norms[i]) for i in selected_idx]
    result_df = pd.DataFrame({"feature": selected_features, "importance_score": selected_scores})
    result_df = result_df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    # save
    out_csv = os.path.join(args.outdir, "selected_features.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"[ok] Saved: {out_csv}")

    # plot (default matplotlib style; no explicit colors)
    plt.figure(figsize=(10, 5))
    plt.bar(result_df["feature"], result_df["importance_score"])
    plt.title("Selected Features by ERCCFS (Row-Norm of W)")
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "selected_features.png")
    plt.savefig(out_png, dpi=200)
    print(f"[ok] Saved: {out_png}")

if __name__ == "__main__":
    main()
