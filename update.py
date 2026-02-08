# ============================================================
# TRAIN + PREDICT LOOP (FINAL VERSION)
# ============================================================

for t in targets:

    print(f"\n🔹 Processing target: {t}")

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if t not in train_df.columns:
        print(f"[SKIP] {t} not found")
        continue

    mask = train_df[t].notna()
    n_rows = int(mask.sum())

    if n_rows < MIN_ROWS_PER_TARGET:
        print(f"[SKIP] {t} too few rows ({n_rows})")

        summary_rows.append({
            "target": t,
            "status": "skip_too_few_rows",
            "n_rows": n_rows
        })
        continue

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    X_train_t = align_features(
        train_df.loc[mask],
        feature_cols_cleaned
    )

    y_train_t = train_df.loc[mask, t].copy()

    # Ordinal encoding
    y_code, label_meta = make_ordinal_codes(y_train_t)
    K = int(label_meta["n_classes"])

    if K < 2:
        print(f"[SKIP] {t} single class")

        summary_rows.append({
            "target": t,
            "status": "skip_single_class",
            "n_rows": n_rows
        })
        continue

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = compute_class_weights(y_code)
    class_weights = adjust_extreme_weights(
        class_weights,
        boost=1.5
    )

    sample_weights = pd.Series(y_code)\
        .map(class_weights)\
        .values

    print(f"[INFO] Class weights: {class_weights}")

    # --------------------------------------------------------
    # Train CORAL model
    # --------------------------------------------------------

    try:

        model = LGBMCoralModel(
            num_classes=K,
            config=coral_cfg,
            decode_grid=decode_grid
        )

        model.fit(
            X_train_t,
            y_code,
            sample_weight=sample_weights,
            tune_weights=True
        )

    except Exception as e:

        print(f"[ERROR] Training failed for {t}: {e}")

        summary_rows.append({
            "target": t,
            "status": "error_fit",
            "error": repr(e)
        })

        continue

    # --------------------------------------------------------
    # Decode tuning (TRAIN)
    # --------------------------------------------------------

    cum_probas_train = model.predict_cumproba(
        X_train_t
    )

    thresholds, best_mae = tune_decode_thresholds(
        y_true=y_code,
        cum_probas=cum_probas_train,
        grid=decode_grid
    )

    pred_code_train = decode_with_thresholds(
        cum_probas_train,
        thresholds
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_code,
        pred_code_train
    )

    tail_recall = recall_score(
        y_code,
        pred_code_train,
        labels=[0, K - 1],
        average="macro"
    )

    print(f"[METRIC] MAE: {mae:.4f}")
    print(f"[METRIC] Tail recall: {tail_recall:.4f}")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm_dir = Path(MODELS_DIR)
    cm_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_code,
        pred_code_train
    )

    pd.DataFrame(cm).to_csv(
        cm_dir / f"{t}_confusion_matrix.csv",
        index=False
    )

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{t} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        cm_dir / f"{t}_confusion_matrix.png"
    )

    plt.close()

    print(f"[OK] Confusion matrix saved for {t}")

    # --------------------------------------------------------
    # FULL DATASET PREDICTION
    # --------------------------------------------------------

    pred_code_full = model.predict(X_full)

    # Decode back to original labels
    pred_label_full = decode_to_original_labels(
        pred_code_full,
        label_meta["from_code"]
    )

    # ✅ GUARANTEED OUTPUT COLUMN
    out_df[f"{t}_pred"] = pred_label_full

    # Optional code column
    out_df[f"{t}_pred_code"] = pred_code_full

    # --------------------------------------------------------
    # Save artifact
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "thresholds": thresholds,
        "class_weights": class_weights,
        "mae": float(mae),
        "tail_recall": float(tail_recall),
        "label_meta": label_meta
    }

    joblib.dump(
        artifact,
        Path(MODELS_DIR) /
        f"{t}_coral_artifact.joblib"
    )

    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------

    summary_rows.append({
        "target": t,
        "status": "trained_and_predicted",
        "n_rows": n_rows,
        "n_classes": K,
        "mae": float(mae),
        "tail_recall": float(tail_recall)
    })
