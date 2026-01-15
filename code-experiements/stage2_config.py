from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Stage2Config:
    # -----------------------------
    # MODE
    # -----------------------------
    MODEL_TYPE: str = "predictassessment"  # "predictassessment" | "predictbudget" | "assessmentbudget"
    DEV_ROW_SUBSET: int | None = None
    DEV_ROW_SUBSET_MODE: str = "head"  # "head" or "sample"

    CLASS_MODEL_TYPES: tuple[str, ...] = ("predictassessment",)
    REGRESSION_MODEL_TYPES: tuple[str, ...] = ("predictbudget", "assessmentbudget")

    # -----------------------------
    # TARGET SELECTION
    # -----------------------------
    TARGETS_INCLUDE: Optional[List[str]] = None  # exact names, order preserved
    TARGETS_EXCLUDE: List[str] = field(default_factory=list)

    # -----------------------------
    # PREFIXES
    # -----------------------------
    ID_PREFIX: str = "id_"
    FEATURE_PREFIX: str = "ft_"
    TARGET_PREFIX: str = "y_"
    BUDGET_PREFIX: str = "budget_"

    # -----------------------------
    # TRAINING RULES
    # -----------------------------
    RANDOM_STATE: int = 42
    MIN_SAMPLES_PER_TARGET: int = 200
    MIN_CLASS_COUNT_FOR_TRAINING: int = 2
    MAX_N_SPLITS_CLASSIFICATION: int = 5
    MAX_N_SPLITS_REGRESSION: int = 5

    FEATURE_REDUCTION_TOP_N_FEATURES: int = 50

    USE_CATBOOST_ENCODER: bool = True
    CAT_FILL_VALUE: str = "NA_CAT"

    # only set if you must
    NUM_IMPUTE: str | None = None  # "median" or None

    # decoded label sanity-check sample size stored in results
    LABEL_EXAMPLE_N: int = 100

    # -----------------------------
    # PARALLELISM
    # -----------------------------
    CPU_COUNT: int = os.cpu_count() or 4
    MAX_TARGET_JOBS: int = 16

    # -----------------------------
    # MLFLOW (optional)
    # -----------------------------
    USE_MLFLOW: bool = False
    MLFLOW_EXPERIMENT_NAME: str = "stage2_model_training"

    @property
    def TASK_MODE(self) -> str:
        if self.MODEL_TYPE in self.CLASS_MODEL_TYPES:
            return "classification"
        if self.MODEL_TYPE in self.REGRESSION_MODEL_TYPES:
            return "regression"
        raise ValueError(f"Invalid MODEL_TYPE={self.MODEL_TYPE}")

    @property
    def ENABLED_MODELS(self) -> List[str]:
        if self.TASK_MODE == "regression":
            return ["RF_REG"]
        return ["RF", "LGBM", "XGB", "HGB", "CB"]

    @property
    def N_JOBS_TARGETS(self) -> int:
        return max(min(self.CPU_COUNT - 1, self.MAX_TARGET_JOBS), 2)

    # -----------------------------
    # PATHS (Stage-2 outputs)
    # -----------------------------
    @property
    def BASE_DIR(self) -> Path:
        return Path(f"ae_models_pipeline/{self.MODEL_TYPE}")

    @property
    def FEATURE_IMPORTANCE_DIR(self) -> Path:
        # Stage-1 output folder that Stage-2 consumes
        return self.BASE_DIR / "feature_importances"

    @property
    def TRAINED_MODELS_DIR(self) -> Path:
        return self.BASE_DIR / "trained_models"

    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"

    @property
    def SKIPPED_TARGETS_CSV(self) -> Path:
        return self.TRAINED_MODELS_DIR / "skipped_targets_stage2.csv"

    @property
    def CV_RESULTS_CSV(self) -> Path:
        return self.TRAINED_MODELS_DIR / "model_cv_results_parallel.csv"

    @property
    def CV_RESULTS_JSON(self) -> Path:
        return self.TRAINED_MODELS_DIR / "model_cv_results_parallel.json"

    def filter_targets(self, y_cols: List[str]) -> List[str]:
        out = list(y_cols)

        if self.TARGETS_INCLUDE is not None:
            inc = set(self.TARGETS_INCLUDE)
            out = [y for y in out if y in inc]

        if self.TARGETS_EXCLUDE:
            exc = set(self.TARGETS_EXCLUDE)
            out = [y for y in out if y not in exc]

        if self.TARGETS_INCLUDE is not None:
            return [y for y in self.TARGETS_INCLUDE if y in out]

        return out
