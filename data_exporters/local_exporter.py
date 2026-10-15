"""Data exporter block: persist metrics, importances, and the stats summary."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from utils.config import OUTPUTS
from utils.logging_config import get_logger
logger = get_logger(__name__)

def export_results(metrics: pd.DataFrame, fi: pd.DataFrame,
                   stats_summary: Dict[str, Any]) -> Path:
    """Write metrics (CSV+JSON), feature importances (CSV), and stats (JSON)."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    (OUTPUTS / "model_comparison.json").write_text(
        json.dumps(metrics.to_dict(orient="records"), indent=2))
    fi.head(20).to_csv(OUTPUTS / "rf_feature_importance.csv", index=False)
    (OUTPUTS / "results_summary.json").write_text(json.dumps(stats_summary, indent=2))
    logger.info("Wrote results to %s", OUTPUTS)
    return OUTPUTS / "model_comparison.csv"
