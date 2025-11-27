# run_multitarget_experiment.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    setup_mlflow,
    get_models_and_spaces,
    RESULTS_CSV_PATH,
    AGG_PLOT_PATH,
)
from data_preprocessing import load_data, split_features_targets
from model_training import tune_and_train_one_model_for_target


def main():
    # 1) MLflow config
    setup_mlflow()

    # 2) Load data
    df = load_data()

    # 3) Split features / targets
    X, target_cols, id_cols = split_features_targets(df, target_prefix="y_")

    print("Targets:", target_cols)
    print("Feature columns:", list(X.columns)[:10], "...")
    if id_cols:
        print("ID columns:", id_cols)

    # 4) Models + hyperparameter spaces
    models_and_spaces = get_models_and_spaces()

    # 5) Loop over targets & models
    all_rows = []

    for target_col in target_cols:
        print(f"\n=== Processing target: {target_col} ===")

        # Skip if too few non-missing labels
        n_non_missing = df[target_col].notna().sum()
        if n_non_missing < 50:
            print(f"  Skipping {target_col}: only {n_non_missing} labeled rows.")
            continue

        for model_name, cfg in models_and_spaces.items():
            print(f"\n  --> Training {model_name} for {target_col}")
            row = tune_and_train_one_model_for_target(
                model_name=model_name,
                base_estimator=cfg["estimator"],
                param_distributions=cfg["param_distributions"],
                df=df,
                X_full=X,
                target_col=target_col,
            )
            all_rows.append(row)

    # 6) Save metrics to CSV
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved metrics to {RESULTS_CSV_PATH}")
    print(results_df.head())

    # 7) Aggregate & plot per-model performance across targets (using CV F1 macro mean)
    metric_col = "cv_" + target_cols[0] + "_RandomForest_f1_macro_mean"  # just to inspect; but targets differ
    # Instead, we’ll compute model-level average of all available cv_*_f1_macro_mean columns.

    # Collect all cv f1_macro metrics and average by model
    f1_cols = [c for c in results_df.columns if c.endswith("f1_macro_mean")]
    # Reshape: each row has multiple f1_macro_mean (one per target); we'll average across them per model
    agg_rows = []
    for model in results_df["model"].unique():
        sub = results_df[results_df["model"] == model]
        # take all f1_macro_mean columns and average their values
        f1_values = sub[f1_cols].values.flatten()
        f1_values = f1_values[~np.isnan(f1_values)]
        agg_rows.append({"model": model, "mean_cv_f1_macro": f1_values.mean() if len(f1_values) else np.nan})

    agg_df = pd.DataFrame(agg_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agg_df["model"], agg_df["mean_cv_f1_macro"])
    ax.set_ylabel("Mean CV F1 Macro (across targets)")
    ax.set_title("Model Comparison across Multi-Targets")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(AGG_PLOT_PATH, dpi=150)
    plt.close()

    print(f"Saved aggregate comparison plot to {AGG_PLOT_PATH}")


if __name__ == "__main__":
    main()
