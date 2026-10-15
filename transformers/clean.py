"""Transformer block: finalise the analytical dataset."""
from __future__ import annotations
import pandas as pd
from utils.config import TARGET
from utils.logging_config import get_logger
logger = get_logger(__name__)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop null targets/years, coerce dtypes; return the analytical frame."""
    out = df[df[TARGET].notnull()].copy()
    out["inspection_completed_date"] = pd.to_datetime(
        out["inspection_completed_date"], errors="coerce")
    out = out.dropna(subset=["inspection_year", TARGET]).copy()
    out["inspection_year"] = out["inspection_year"].astype(int)
    out[TARGET] = out[TARGET].astype(int)
    logger.info("Analytical dataset: %d rows, failure rate %.4f", len(out), out[TARGET].mean())
    return out
