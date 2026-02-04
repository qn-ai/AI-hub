import numpy as np
import pandas as pd

def _psi_numeric(train: pd.Series, unseen: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index for numeric features.
    Uses quantile bins from train to be robust.
    """
    tr = train.replace([np.inf, -np.inf], np.nan).dropna()
    un = unseen.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 50 or len(un) < 50:
        return 0.0

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(tr.quantile(qs).values)
    if len(edges) <= 2:
        return 0.0

    tr_bin = pd.cut(tr, bins=edges, include_lowest=True).value_counts(normalize=True)
    un_bin = pd.cut(un, bins=edges, include_lowest=True).value_counts(normalize=True)

    # align
    tr_bin, un_bin = tr_bin.align(un_bin, fill_value=0.0)

    eps = 1e-6
    tr_p = np.clip(tr_bin.values, eps, 1.0)
    un_p = np.clip(un_bin.values, eps, 1.0)

    return float(np.sum((un_p - tr_p) * np.log(un_p / tr_p)))


def stage0_stability_filter(
    X_train: pd.DataFrame,
    X_unseen: pd.DataFrame,
    candidate_features: list[str],
    *,
    miss_unseen_max: float = 0.80,
    miss_shift_max: float = 0.30,
    zshift_max: float = 1.5,
    use_psi: bool = False,
    psi_max: float = 0.25,
    psi_bins: int = 10,
):
    """
    Returns: kept_features, drop_report_df
    """
    feats = [f for f in candidate_features if f in X_train.columns]
    # ensure unseen has columns
    for f in feats:
        if f not in X_unseen.columns:
            X_unseen[f] = np.nan

    miss_tr = X_train[feats].isna().mean()
    miss_un = X_unseen[feats].isna().mean()
    miss_shift = (miss_un - miss_tr).abs()

    # numeric stats (coerce to numeric where possible)
    Xt = X_train[feats].copy()
    Xu = X_unseen[feats].copy()

    for f in feats:
        if Xt[f].dtype == "object":
            Xt[f] = pd.to_numeric(Xt[f], errors="coerce")
        if Xu[f].dtype == "object":
            Xu[f] = pd.to_numeric(Xu[f], errors="coerce")

    mean_tr = Xt.mean(numeric_only=True)
    std_tr = Xt.std(numeric_only=True).replace(0, np.nan)
    mean_un = Xu.mean(numeric_only=True)

    zshift = ((mean_un - mean_tr).abs() / std_tr).reindex(feats).fillna(0.0)

    psi = pd.Series(0.0, index=feats)
    if use_psi:
        for f in feats:
            if pd.api.types.is_numeric_dtype(Xt[f]):
                psi[f] = _psi_numeric(Xt[f], Xu[f], bins=psi_bins)

    drop_reasons = []
    for f in feats:
        reasons = []
        if miss_un[f] > miss_unseen_max:
            reasons.append(f"miss_unseen>{miss_unseen_max}")
        if miss_shift[f] > miss_shift_max:
            reasons.append(f"miss_shift>{miss_shift_max}")
        if zshift[f] > zshift_max:
            reasons.append(f"zshift>{zshift_max}")
        if use_psi and psi[f] > psi_max:
            reasons.append(f"psi>{psi_max}")
        drop_reasons.append("|".join(reasons))

    report = pd.DataFrame({
        "feature": feats,
        "miss_train": miss_tr.values,
        "miss_unseen": miss_un.values,
        "miss_shift_abs": miss_shift.values,
        "zshift": zshift.values,
        "psi": psi.values if use_psi else np.zeros(len(feats)),
        "drop_reasons": drop_reasons,
    })

    kept = report[report["drop_reasons"] == ""]["feature"].tolist()
    dropped = report[report["drop_reasons"] != ""]

    return kept, report.sort_values(
        by=["drop_reasons", "miss_shift_abs", "zshift"],
        ascending=[False, False, False]
    )



# Load unseen just for stability filtering (no labels needed)
df_unseen = pd.read_csv(config.UNSEEN_CSV)

# Build unseen feature frame with same raw feature columns
# (if some missing, they stay NaN, which is what we want here)
X_unseen_raw = df_unseen.copy()
for c in feature_cols:  # original feature cols before Stage-0
    if c not in X_unseen_raw.columns:
        X_unseen_raw[c] = np.nan

X_unseen_raw = X_unseen_raw[feature_cols]  # keep same order
X_unseen_raw = X_unseen_raw.replace([np.inf, -np.inf], np.nan)

# Apply stability filter on Stage-0 selected features
sel_stable, stability_report = stage0_stability_filter(
    X_train=X_tr[sel],
    X_unseen=X_unseen_raw[sel],
    candidate_features=sel,
    miss_unseen_max=0.80,
    miss_shift_max=0.30,
    zshift_max=1.5,
    use_psi=False,      # start False; turn True later if needed
)

print(f"[Stage0b] selected={len(sel)} stable={len(sel_stable)} dropped={len(sel)-len(sel_stable)}")
stability_report.to_csv(f"{config.STAGE0_REPORT_DIR}/stability_report.csv", index=False)

# Replace your feature list with stable features
sel = sel_stable



def go_no_go(train_pct, pred_pct, drift_l1, top_class_pct):
    K = len(train_pct)
    # rare defined as >=1% in training (business-relevant)
    rare_business = [k for k in range(K) if train_pct[k] >= 0.01]
    collapsed = [k for k in rare_business if pred_pct[k] == 0.0]
    spiked = [k for k in range(K) if train_pct[k] > 0 and pred_pct[k] > 3.0 * train_pct[k]]

    hard_fail = (
        (top_class_pct >= 0.90) or
        (drift_l1 >= 0.35) or
        (len(collapsed) > 0)
    )

    warn = (
        (0.80 <= top_class_pct < 0.90) or
        (0.25 <= drift_l1 < 0.35) or
        (len(spiked) > 0)
    )

    if hard_fail:
        decision = "NO-GO"
    elif warn:
        decision = "REVIEW"
    else:
        decision = "GO"

    return decision, collapsed, spiked


def add_exec_summary_page(pdf, results_df, meta: dict):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")

    # counts
    counts = results_df["decision"].value_counts().to_dict()
    go = counts.get("GO", 0)
    review = counts.get("REVIEW", 0)
    nogo = counts.get("NO-GO", 0)

    worst_drift = results_df.sort_values("drift_l1", ascending=False).head(10)
    worst_dom = results_df.sort_values("top_class_pct", ascending=False).head(10)

    lines = []
    lines.append("MODEL GOVERNANCE SUMMARY (Unseen Prediction Run)")
    lines.append("")
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append(f"Targets: {len(results_df)}")
    lines.append(f"GO: {go}   |   REVIEW: {review}   |   NO-GO: {nogo}")
    lines.append("")
    lines.append("Top 10 by distribution drift (L1):")
    for _, r in worst_drift.iterrows():
        lines.append(f"  - {r['target']}: drift={r['drift_l1']:.3f}, top={r['top_class']}@{r['top_class_pct']*100:.1f}%, decision={r['decision']}")
    lines.append("")
    lines.append("Top 10 by dominance:")
    for _, r in worst_dom.iterrows():
        lines.append(f"  - {r['target']}: top={r['top_class']}@{r['top_class_pct']*100:.1f}%, drift={r['drift_l1']:.3f}, decision={r['decision']}")

    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
