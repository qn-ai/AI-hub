import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# =====================================================
DATA_PATH = "input_data.csv"

MODELS = {
    "RF": "Random Forest",
    "LGBM": "LightGBM",
    "CAT": "CatBoost"
}

MIN_LABELS = 200
IMBALANCE_THRESHOLD = 0.80
FOLD_CANDIDATES = [3, 5, 10]
N_JOBS = -1
SHAP_SAMPLE = 500
# =====================================================


def train_model(model_name, X, y):
    """Builds the appropriate model instance."""
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        )

    elif model_name == "LGBM":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "CAT":
        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss" if y.nunique()==2 else "MultiClass",
            verbose=False
        )


def process_target(df, target, ft_cols, model_key):

    model_name = MODELS[model_key]
    print(f"\n=== {model_name} | Target {target} ===")

    mask = df[target].notna()
    y = df.loc[mask, target]
    X = df.loc[mask, ft_cols]
    X_full = df[ft_cols]

    if y.nunique() < 2 or len(y) < MIN_LABELS:
        return None

    # imbalance detection
    class_dist = y.value_counts(normalize=True)
    maj_prop = class_dist.max()
    is_imbalanced = maj_prop >= IMBALANCE_THRESHOLD

    # best CV fold
    fold_scores = {}
    for k in FOLD_CANDIDATES:
        model_temp = train_model(model_key, X, y)
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model_temp, X, y, cv=cv, scoring="accuracy")
        fold_scores[k] = (scores.mean(), scores.std())

    best_k = max(fold_scores, key=lambda k: fold_scores[k][0])
    best_cv_mean, best_cv_std = fold_scores[best_k]

    # final model
    model = train_model(model_key, X, y)
    model.fit(X, y)

    # predictions
    y_pred = model.predict(X_full)
    y_proba = model.predict_proba(X_full)
    conf = y_proba[:, 1] if y.nunique()==2 else y_proba.max(axis=1)

    # metrics
    acc = accuracy_score(y, model.predict(X))
    f1 = f1_score(y, model.predict(X), average="macro")

    # SHAP
    shap_path = f"shap_{model_key}_{target}.png"
    try:
        sample_X = X.sample(min(len(X), SHAP_SAMPLE), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_X)

        shap.summary_plot(
            shap_values if y.nunique()==2 else shap_values[0],
            sample_X,
            show=False
        )
        plt.tight_layout()
        plt.savefig(shap_path, dpi=120)
        plt.close()
    except Exception as e:
        print(f"SHAP error: {e}")

    # return everything
    metrics_row = {
        "target": target,
        "n_labelled": len(y),
        "cv_mean_acc": best_cv_mean,
        "cv_std_acc": best_cv_std,
        "best_cv_folds": best_k,
        "accuracy_train": acc,
        "f1_train": f1,
        "is_imbalanced": is_imbalanced,
        "majority_class_prop": maj_prop
    }

    predictions = pd.DataFrame({
        target + f"_{model_key}_pred": y_pred,
        target + f"_{model_key}_conf": conf
    })

    return metrics_row, predictions



def main():

    df = pd.read_csv(DATA_PATH)

    id_cols = [c for c in df.columns if c.startswith("id_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    ft_cols = [c for c in df.columns if c.startswith("ft_")]

    for model_key in MODELS.keys():

        print(f"\n==============================")
        print(f"   TRAINING MODEL: {MODELS[model_key]}")
        print(f"==============================")

        metrics_all = []
        result_table = df[id_cols + y_cols].copy()

        processed = Parallel(n_jobs=N_JOBS)(
            delayed(process_target)(df, target, ft_cols, model_key)
            for target in y_cols
        )

        for result in processed:
            if result is None:
                continue
            metrics_row, preds_df = result

            metrics_all.append(metrics_row)
            result_table = pd.concat([result_table, preds_df], axis=1)

        # save outputs
        out_table_path = f"output_table_{model_key}.csv"
        out_metrics_path = f"metrics_{model_key}.csv"

        result_table.to_csv(out_table_path, index=False)
        pd.DataFrame(metrics_all).to_csv(out_metrics_path, index=False)

        print(f"\nSaved: {out_table_path}")
        print(f"Saved: {out_metrics_path}")

    print("\nALL MODELS COMPLETE ✓")


if __name__ == "__main__":
    main()



import streamlit as st
import pandas as pd
import os

# ================================
# Load all three model outputs
# ================================

@st.cache_data
def load_all():
    tables = {
        "RF": pd.read_csv("output_table_RF.csv"),
        "LGBM": pd.read_csv("output_table_LGBM.csv"),
        "CAT": pd.read_csv("output_table_CAT.csv")
    }
    metrics = {
        "RF": pd.read_csv("metrics_RF.csv"),
        "LGBM": pd.read_csv("metrics_LGBM.csv"),
        "CAT": pd.read_csv("metrics_CAT.csv")
    }
    return tables, metrics

tables, metrics = load_all()

models = {"RF": "Random Forest", "LGBM": "LightGBM", "CAT": "CatBoost"}

# ================================
# SIDEBAR
# ================================

st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio(
    "Go to Page",
    ["Overview", "Predictions Explorer", "Model Comparison", "SHAP Viewer"]
)

model_choice = st.sidebar.selectbox(
    "Model:",
    list(models.keys()),
    format_func=lambda x: models[x]
)


# ================================
# PAGE 1 — OVERVIEW
# ================================

if page == "Overview":
    st.title("📊 Multi-Model Dashboard")
    st.write("Models included:")
    st.write("- Random Forest")
    st.write("- LightGBM")
    st.write("- CatBoost")

    st.subheader("Dataset Summary")
    st.write("Rows:", len(tables["RF"]))
    st.write("Targets:", len([c for c in tables["RF"].columns if c.startswith("y_")]))

    st.subheader("Metrics Preview")
    show_model = st.selectbox("Select model to preview", models.keys())
    st.dataframe(metrics[show_model].head())


# ================================
# PAGE 2 — PREDICTIONS EXPLORER
# ================================

if page == "Predictions Explorer":
    st.title("🔍 Predictions Explorer")

    table = tables[model_choice]

    id_cols = [c for c in table.columns if c.startswith("id_")]
    y_cols = [c for c in table.columns if c.startswith("y_") and not "_pred" in c and not "_conf" in c]

    id_col = st.selectbox("ID column:", id_cols)
    selected_target = st.selectbox("Target:", y_cols)

    pred_col = selected_target + f"_{model_choice}_pred"
    conf_col = selected_target + f"_{model_choice}_conf"

    search = st.text_input("Search ID")

    df = table.copy()

    if search != "":
        df = df[df[id_col].astype(str) == search]

    st.dataframe(df[[id_col, selected_target, pred_col, conf_col]].head(100))


# ================================
# PAGE 3 — MODEL COMPARISON
# ================================

if page == "Model Comparison":
    st.title("📈 Model Comparison")

    all_metrics = []
    for key in models.keys():
        m = metrics[key].copy()
        m["model"] = models[key]
        all_metrics.append(m)

    merged = pd.concat(all_metrics)

    st.dataframe(merged)

    st.subheader("Accuracy Comparison")
    st.bar_chart(merged.pivot(index="target", columns="model", values="cv_mean_acc"))

    st.subheader("F1-score Comparison")
    st.bar_chart(merged.pivot(index="target", columns="model", values="f1_train"))


# ================================
# PAGE 4 — SHAP VIEWER
# ================================

if page == "SHAP Viewer":
    st.title("🔍 SHAP Explanations")

    target_list = metrics[model_choice]["target"].tolist()
    target = st.selectbox("Select target:", target_list)

    shap_path = f"shap_{model_choice}_{target}.png"

    if os.path.exists(shap_path):
        st.image(shap_path, caption=f"SHAP — Model={models[model_choice]}, Target={target}")
    else:
        st.warning("SHAP file not found.")
