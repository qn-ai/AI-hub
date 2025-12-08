
# Multi-Target Feature Selection, Model Training & Scoring Pipeline

A complete, production-grade pipeline for **multi-target classification** on wide, sparse datasets with:

- Hundreds of mixed-type features (`ft_*`)
- Dozens to hundreds of targets (`y_*`)
- Significant missingness
- Per-target feature selection
- Parallel model training and prediction
- Large dataset scoring (e.g., 377k rows)
- Optional MLflow tracking
- Full logging, including per-target logs

---

# 1. Project Structure

```
.
├── input_data.csv
├── new_data.csv
├── compute_feature_importances.py       # Stage-1
├── train_stage2.py                      # Stage-2
├── predict_stage3.py                    # Stage-3
├── feature_importances/                 # per-target feature rankings
├── trained_models/                      # per-target best model .joblib
├── logs/                                # global + per-target logs
├── model_cv_results_parallel.csv        # Stage-2 metrics
├── stage3_predictions.csv               # final scoring output
└── README.md
```

---

# 2. Data Schema Requirements

| Prefix | Description |
|--------|-------------|
| `id_`  | Identifier columns (e.g., `id_pwd_id`) |
| `ft_`  | Feature columns (numeric or categorical) |
| `y_`   | Target columns (binary or multiclass) |

Example:

```
id_pwd_id, ft_income, ft_gender, ft_state, y_default, y_risk_group, ...
```

---

# 3. Stage-1 — Feature Importance Pipeline

(Full content omitted here to keep the example concise — your actual README should include the entire section.)

---

# 4. Stage-2 — Model Training Pipeline

(Full content omitted — insert full version.)

---

# 5. Stage-3 — Scoring / Predictions (New Dataset)

(Full content omitted — insert full version.)

---

# 6. Quick Configuration Tips

(Full content omitted — insert full version.)

---

# 7. Workflow (All 3 Stages)

```bash
python compute_feature_importances.py
python train_stage2.py
python predict_stage3.py
```

Or:

```bash
make all
```

---

# 8. Makefile Integration

(Full content omitted — insert full version.)

---

# 9. Troubleshooting

(Full content omitted — insert full version.)

---

# 10. Credits

Uses scikit-learn, LightGBM, CatBoost, XGBoost, joblib, tqdm, MLflow.
