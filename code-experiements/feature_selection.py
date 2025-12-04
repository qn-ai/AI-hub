# feature_selection.py
from pathlib import Path
from typing import List, Tuple

import logging
import pandas as pd

from config import (
    ID_PREFIX,
    FEATURE_PREFIX,
    TARGET_PREFIX,
    FEATURE_IMPORTANCE_DIR,
)

log = logging.getLogger(__name__)


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Return (id_cols, ft_cols, y_cols) based on prefixes."""
    id_cols = [c for c in df.columns if c.startswith(ID_PREFIX)]
    ft_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    y_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    return id_cols, ft_cols, y_cols


def load_selected_features_for_target(y_col: str, ft_cols_all: List[str]) -> List[str]:
    """Select features for a target based on combined importance CSV files.

    Logic:

    1. List ALL *.csv in FEATURE_IMPORTANCE_DIR.
    2. Drop any file whose name ends with:
           _CB.csv, _XGB.csv, _RF.csv, _LGBM.csv
       (those are per-model importance files, not combined).
    3. Among the remaining files, keep only those whose filename contains y_col.
    4. For the matching file(s), pick the largest one (combined importances).
    5. Read that file; it has:
           (blank first column) as feature names,
           RF, LGBM, CB, XGB, mean_rank, ...
       We load with index_col=0 so the blank column becomes the index = feature name.
    6. Keep only rows (features) where:
           RF > 0 AND LGBM > 0 AND CB > 0 AND XGB > 0
    7. Filter that list down to features actually present in ft_cols_all.
    """

    # 1) All CSVs in the dir
    all_csvs: List[Path] = list(FEATURE_IMPORTANCE_DIR.glob("*.csv"))
    if not all_csvs:
        log.warning("No feature importance CSV files found in %s", FEATURE_IMPORTANCE_DIR)
        return []

    # 2) Exclude per-model suffixes
    excluded_suffixes = ("_CB.csv", "_XGB.csv", "_RF.csv", "_LGBM.csv")
    combined_files = [
        p for p in all_csvs
        if not p.name.endswith(excluded_suffixes)
    ]

    if not combined_files:
        log.warning(
            "All feature importance files look like per-model files; "
            "no combined file available for any target."
        )
        return []

    # 3) Filter to files containing this y_col in the name
    matching_files = [p for p in combined_files if y_col in p.name]
    if not matching_files:
        log.warning(
            "No combined feature importance file found for target %s "
            "(looked for filenames containing '%s')",
            y_col,
            y_col,
        )
        return []

    # 4) Pick the largest matching CSV (most complete)
    file_path = max(matching_files, key=lambda p: p.stat().st_size)
    log.info("Using feature importance file for %s: %s", y_col, file_path)

    # 5) Read file: blank col as index = feature names
    fi_df = pd.read_csv(file_path, index_col=0)

    required_cols = ["RF", "LGBM", "CB", "XGB"]
    missing = [c for c in required_cols if c not in fi_df.columns]
    if missing:
        log.warning(
            "Importance file %s for %s is missing required columns %s",
            file_path,
            y_col,
            missing,
        )
        return []

    # 6) Strict rule: ALL four > 0
    mask = (fi_df[required_cols] > 0).all(axis=1)
    selected = fi_df.index[mask].tolist()  # index = feature names

    # 7) Keep only features that exist in dataset ft_ columns
    selected = [f for f in selected if f in ft_cols_all]

    log.info(
        "Target %s: %d features selected (RF, LGBM, CB, XGB > 0) from %s",
        y_col,
        len(selected),
        file_path.name,
    )
    return selected
