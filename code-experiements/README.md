flowchart TB

%% --- Lanes ---
subgraph DATA["Data Input"]
    A1[Load input_data.csv]
    A2[Detect id_, ft_, y_ columns]
end

subgraph S1["Stage-1: Feature Importances"]
    S1A[Filter rows per y_*]
    S1B[Skip rare/degenerate classes]
    S1C[Global cleanup: variance + correlation]
    S1D[Per-target missing cleanup]
    S1E[Build numeric + CatBoost feature views]
    S1F[Train 5 models: RF, LGBM, XGB, HGB, CB]
    S1G[Extract importances]
    S1H[Save feature_importances_<y>.csv]
end

subgraph S2["Stage-2: Model Training"]
    S2A[Load feature_importances_<y>.csv]
    S2B[Select features RF,LGBM,CB,XGB,HGB > 0]
    S2C[Filter non-missing rows]
    S2D[Auto-detect class count + choose CV folds]
    S2E[Skip bad/rare targets]
    S2F[Build numeric + CatBoost views]
    S2G[Cross-validate 5 models]
    S2H[Compute metrics F1/Recall/Accuracy/AUC]
    S2I[Pick best model by F1]
    S2J[Retrain ALL 5 models on full data]
    S2K[Save models: RF/LGBM/XGB/HGB/CB + best.joblib]
    S2L[Append to model_cv_results_parallel.(csv/json)]
end

subgraph S3["Stage-3: Prediction"]
    S3A[Load new dataset]
    S3B{Prediction Mode?}
    S3C1[Mode=Best → Load best.joblib]
    S3C2[Mode=All → Load RF,LGBM,XGB,HGB,CB]
    S3D[Compute predictions + confidence metrics]
    S3E[Save stage3_predictions.csv]
end

subgraph OUTPUT["Outputs"]
    O1[feature_importances/*.csv]
    O2[trained_models/*.joblib]
    O3[model_cv_results_parallel.csv/json]
    O4[stage3_predictions.csv]
    O5[skipped_targets_stage1.csv / stage2.csv]
end

%% --- Links ---
A1 --> A2 --> S1A

S1A --> S1B --> S1C --> S1D --> S1E --> S1F --> S1G --> S1H
S1H --> S2A

S2A --> S2B --> S2C --> S2D --> S2E --> S2F --> S2G --> S2H --> S2I --> S2J --> S2K --> S2L
S2K --> S3A

S3A --> S3B
S3B -->|best| S3C1
S3B -->|all models| S3C2
S3C1 --> S3D --> S3E
S3C2 --> S3D --> S3E

%% Outputs
S1H --> O1
S2K --> O2
S2L --> O3
S3E --> O4
S1B --> O5
S2E --> O5
