"""Data loader block: read the cleaned Calgary inspections CSV."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from utils.config import DATA_CSV
from utils.logging_config import get_logger
logger = get_logger(__name__)

def load_inspections(path: Path = DATA_CSV) -> pd.DataFrame:
    """Return the raw cleaned-inspections table."""
    df = pd.read_csv(path)
    logger.info("Loaded inspections: %s", df.shape)
    return df
