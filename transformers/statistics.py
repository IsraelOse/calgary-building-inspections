"""
Transformer block: the zone-vs-outcome association test.

Tests whether inspection outcome depends on geographic zone, reporting the
chi-square statistic, p-value, and Cramer's V effect size.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import stats
from utils.config import TARGET
from utils.logging_config import get_logger
logger = get_logger(__name__)

def zone_association(df: pd.DataFrame) -> Dict[str, Any]:
    """Chi-square test of independence between zone and failure, with Cramer's V."""
    contingency = pd.crosstab(df["zone"], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))
    result = {"chi2": round(float(chi2), 2), "p_value": float(p),
              "dof": int(dof), "cramers_v": round(cramers_v, 4)}
    logger.info("Zone association: chi2=%.1f p=%.2e Cramers_V=%.4f (effect size)",
                chi2, p, cramers_v)
    return result
