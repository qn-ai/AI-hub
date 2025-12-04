# train_all.py
import json
import time
from typing import List, Dict, Any

import pandas as pd
from joblib import Parallel, delayed

from config import (
    DATA_PATH,
    N_JOBS_TARGETS,
    RESULTS_CSV_PATH,
    RESULTS_JSON_PATH,
    RESOURCE_SUMMARY_PATH,
    _CPU_COUNT,
)
from feature_selection import detect_columns
from logger import get_logger
from mlflow_utils import init_mlflow
from train_target import train_one_target

log = get_logger(__name__)


def main() -> None:
    init_mlflow("multi_target_training")

    log.info("Loading data from %s", DATA_PATH)
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH, low_memory=False)

    id_cols, ft_cols_all, y_cols = detect_columns(df)
    log.info(
        "Detected %d id_, %d ft_, %d y_ columns",
        len(id_cols),
        len(ft_cols_all),
        len(y_cols),
    )
    log.info(
        "Parallel jobs per target: %d (CPU count: %d)",
        N_JOBS_TARGETS,
        _CPU_COUNT,
    )

    results_per_target: List[List[Dict[str, Any]]] = Parallel(
        n_jobs=N_JOBS_TARGETS
    )(
        delayed(train_one_target)(y_col, df, ft_cols_all)
        for y_col in y_cols
    )

    flat_results: List[Dict[str, Any]] = [
        item for sub in results_per_target for item in sub
    ]

    if flat_results:
        df_results = pd.DataFrame(flat_results)
        df_results.to_csv(RESULTS_CSV_PATH, index=False)
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(flat_results, f, indent=2)
        log.info(
            "Saved CV results to %s and %s",
            RESULTS_CSV_PATH,
            RESULTS_JSON_PATH,
        )
    else:
        log.warning("No models were trained; no results to save.")

    total_wall_time = float(time.perf_counter() - t0)
    summary = {
        "total_wall_time_sec": total_wall_time,
        "targets_trained": int(
            pd.DataFrame(flat_results)["target"].nunique()
        )
        if flat_results
        else 0,
        "cpu_count": int(_CPU_COUNT),
        "n_jobs_targets": int(N_JOBS_TARGETS),
    }
    with open(RESOURCE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Training finished in %.2f seconds", total_wall_time)


if __name__ == "__main__":
    main()
