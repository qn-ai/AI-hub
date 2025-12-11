import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("stage1")



PARAM_BASIC = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}

PARAM_TEST1 = {
    "n_estimators": 600,        # more trees for stable rankings
    "max_depth": 20,            # cap depth to limit impurity bias
    "min_samples_split": 10,
    "min_samples_leaf": 5,      # larger leaves = smoother splits
    "max_features": 0.2,        # ~60 features per split for p=300
    "bootstrap": True,          # needed for OOB & diversity
    "oob_score": True,          # quick internal check
    "random_state": 42,
    "n_jobs": -1
}

PARAM_TEST2 = {
    "n_estimators": 600,        # more trees for stable rankings
    "max_depth": 5,            # cap depth to limit impurity bias
    "min_samples_split": 10,
    "min_samples_leaf": 5,      # larger leaves = smoother splits
    "max_features": 0.2,        # ~60 features per split for p=300
    "random_state": 42,
    "n_jobs": -1
}


USE_PARAMS = PARAM_TEST2
# ---------------------------
#  Read data
# ---------------------------
df = return_master_data(
    # nrows=1000,
    add_budget=True,
    model_type=MODEL_TYPE
)
df = df[df["budget_total"].notnull()]


# ---------------------------
#  Copied preprocessing code
# TODO Confirm Queen is using same
# ---------------------------
id_cols, ft_cols, y_col = detect_columns(df, model_type=MODEL_TYPE)

X_all = df[ft_cols].copy()  # noqa: N806
X_base = global_feature_cleanup(X_all, LOG)
df_t = df[df[y_col].notna()].copy()
y_raw = df_t[y_col]
# y_enc, n_classes, min_class = encode_target_for_importance(y_raw, logger=LOG)

X_t = X_base.loc[df_t.index].copy()  # noqa: N806
X_t = per_target_missing_cleanup(X_t, logger=LOG)  # noqa: N806

X_num, X_cb = prepare_feature_views(X_t)  # noqa: N806


# ---------------------------
# Define regressor
# TODO confirm params Queen and Val are using
# ---------------------------

rf = RandomForestRegressor(
    **USE_PARAMS
)


# ---------------------------
# Compute CV error
# ---------------------------
scoring = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "R2": "r2",
}

cv_results = cross_validate(
    rf,
    X_num,
    y_raw,
    cv=5,
    scoring=scoring,
    n_jobs=-1,
)

metric_means = {}
metric_stds = {}

# Avg across folds
for metric in scoring.keys():
    metric_means[metric] = cv_results[f"test_{metric}"].mean()
    metric_stds[metric] = cv_results[f"test_{metric}"].std()
    metric_means[metric] = np.nanmean(cv_results[f"test_{metric}"])
    metric_stds[metric] = np.nanstd(cv_results[f"test_{metric}"])

# Formatting
def fmt_number(value, decimals=0):
    if np.isnan(value): 
        return ""
    # Thousands separators and fixed decimals
    return f"{value:,.{decimals}f}"

def fmt_percent_from_mape(value, decimals=0):
    if np.isnan(value): 
        return ""
    # MAPE should be positive; convert to percent regardless of input scale
    # If |value| <= 1 treat as ratio (e.g., 0.129 -> 12.90%), else assume already percent
    scaled = abs(value) * 100
    return f"{scaled:.{decimals}f}%"

# Return to df
results_df = pd.DataFrame({
        "model": f"v0 RF {USE_PARAMS}",
        "MAE":   [fmt_number(-1 *metric_means["MAE"])],
        "MSE":   [fmt_number(-1 * metric_means["MSE"])],
        "RMSE":  [fmt_number(np.sqrt(-1 * metric_means["MSE"]))],
        "MAPE":  [fmt_percent_from_mape(-1 * metric_means["MAPE"])],
        "R2":    metric_means["R2"],
})
results_df.to_csv("regressor_error.csv", index=False)
