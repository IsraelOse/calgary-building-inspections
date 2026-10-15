"""
Pipeline orchestration: calgary_inspections.

data_loaders -> transformers (clean -> stats -> features -> split -> model)
-> data_exporters.  Run:  python -m pipelines.calgary_inspections.run
"""
from __future__ import annotations
import pandas as pd
from data_loaders.inspection_loader import load_inspections
from transformers.clean import clean
from transformers.statistics import zone_association
from transformers.features import build_xy, split_xy
from transformers.model_training import train_and_evaluate
from data_exporters.local_exporter import export_results
from utils.logging_config import get_logger
logger = get_logger("pipelines.calgary_inspections")

def run() -> pd.DataFrame:
    logger.info("=== calgary_inspections pipeline start ===")
    df = clean(load_inspections())
    assoc = zone_association(df)
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = split_xy(X, y)
    metrics, fi, extras = train_and_evaluate(X_train, X_test, y_train, y_test)
    summary = {"n_rows": len(df), "overall_failure_rate": round(float(y.mean()), 4),
               "zone_association": assoc, **extras,
               "top_feature": fi.iloc[0]["feature"]}
    export_results(metrics, fi, summary)
    logger.info("=== complete: Cramers_V=%.4f, zone importance %.1f%% ===",
                assoc["cramers_v"], extras["rf_zone_importance_share"] * 100)
    return metrics

if __name__ == "__main__":
    print(run().to_string(index=False))
