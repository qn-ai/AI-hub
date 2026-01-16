from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Stage3Config:
    # -----------------------------
    # MODE
    # -----------------------------
    MODEL_TYPE: str = "predictassessment"  # predictassessment | predictbudget | assessmentbudget
    DEV_ROW_SUBSET: int | None = None
    DEV_ROW_SUBSET_MODE: str = "head"  # head | sample

    CLASS_MODEL_TYPES: tuple[str, ...] = ("predictassessment",)
    REGRESSION_MODEL_TYPES: tuple[str, ...] = ("predictbudget", "assessmentbudget")

    # -----------------------------
    # TARGET SELECTION
    # -----------------------------
    TARGETS_INCLUDE: Optional[List[str]] = None
    TARGETS_EXCLUDE: List[str] = field(default_factory=list)

    # -----------------------------
    # PREDICTION MODE
    # -----------------------------
    PREDICTION_MODE: str = "best"  # "best" | "all_models"
    ENABLED_MODELS_CLASSIFICATION: List[str] = field(default_factory=lambda: ["RF", "LGBM", "XGB", "HGB", "CB"])
    ENABLED_MODELS_REGRESSION: List[str] = field(default_factory=lambda: ["RF_REG"])

    MODEL_SUFFIX: str = "model4"  # used in output column names

    # -----------------------------
    # INPUTS/OUTPUTS
    # -----------------------------
    NEW_DATA_NAME: str = "new_data.csv"
    OUTPUT_BASENAME: str = "stage3_predictions"

    # -----------------------------
    # PREFIXES
    # -----------------------------
    ID_PREFIX: str = "id_"
    FEATURE_PREFIX: str = "ft_"
    TARGET_PREFIX: str = "y_"
    BUDGET_PREFIX: str = "budget_"

    # -----------------------------
    # PERFORMANCE
    # -----------------------------
    RANDOM_STATE: int = 42
    CHUNK_SIZE: int = 50_000

    CPU_COUNT: int = os.cpu_count() or 4
    MAX_TARGET_JOBS: int = 16

    # -----------------------------
    # FEATURE PREP
    # -----------------------------
    USE_CATBOOST_ENCODER: bool = True
    CAT_FILL_VALUE: str = "NA_CAT"
    FEATURE_REDUCTION_TOP_N_FEATURES: int = 50
    NUM_IMPUTE: str | None = None  # "median" or None

    # -----------------------------
    # OUTPUT OPTIONS
    # -----------------------------
    RETURN_CLASS_PROBABILITY_DICT: bool = True
    RETURN_CONFIDENCE_METRICS: bool = True  # stage2 global metrics written per-row as constants

    # -----------------------------
    # PATHS
    # -----------------------------
    @property
    def TASK_MODE(self) -> str:
        if self.MODEL_TYPE in self.CLASS_MODEL_TYPES:
            return "classification"
        if self.MODEL_TYPE in self.REGRESSION_MODEL_TYPES:
            return "regression"
        raise ValueError(f"Invalid MODEL_TYPE={self.MODEL_TYPE}")

    @property
    def ENABLED_MODELS(self) -> List[str]:
        return self.ENABLED_MODELS_REGRESSION if self.TASK_MODE == "regression" else self.ENABLED_MODELS_CLASSIFICATION

    @property
    def N_JOBS_TARGETS(self) -> int:
        return max(min(self.CPU_COUNT - 1, self.MAX_TARGET_JOBS), 2)

    @property
    def BASE_DIR(self) -> Path:
        return Path(f"ae_models_pipeline/{self.MODEL_TYPE}")

    @property
    def FEATURE_IMPORTANCE_DIR(self) -> Path:
        return self.BASE_DIR / "feature_importances"

    @property
    def TRAINED_MODELS_DIR(self) -> Path:
        return self.BASE_DIR / "trained_models"

    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"

    @property
    def CV_RESULTS_CSV(self) -> Path:
        return self.TRAINED_MODELS_DIR / "model_cv_results_parallel.csv"

    @property
    def CHECKS_OUTPUT_JSON(self) -> Path:
        return self.BASE_DIR / "stage_3_checks.json"

    @property
    def OUTPUT_PATH(self) -> Path:
        return Path(f"{self.OUTPUT_BASENAME}_{self.MODEL_TYPE}_{self.MODEL_SUFFIX}.csv")

    def filter_targets(self, targets: List[str]) -> List[str]:
        out = list(targets)
        if self.TARGETS_INCLUDE is not None:
            inc = set(self.TARGETS_INCLUDE)
            out = [t for t in out if t in inc]
        if self.TARGETS_EXCLUDE:
            exc = set(self.TARGETS_EXCLUDE)
            out = [t for t in out if t not in exc]
        if self.TARGETS_INCLUDE is not None:
            return [t for t in self.TARGETS_INCLUDE if t in out]
        return out
