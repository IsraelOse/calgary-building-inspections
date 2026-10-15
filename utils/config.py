"""Central configuration for the Calgary Building Inspections pipeline."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

SEED: int = 42
TEST_SIZE: float = 0.30

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_CSV: Path = ROOT / "Calgary_Inspection_Clean.csv"
OUTPUTS: Path = ROOT / "outputs"

TARGET: str = "failed"
FEATURES: List[str] = ["zone", "inspection_type_desc", "inspection_year"]
ONEHOT_COLS: List[str] = ["zone", "inspection_type_desc"]

MODEL_PARAMS: Dict[str, dict] = {
    "Logistic Regression": {"max_iter": 2000},
    "Decision Tree": {"max_depth": 5, "min_samples_leaf": 200},
    "Random Forest": {"n_estimators": 200, "max_depth": 12,
                      "min_samples_leaf": 50, "n_jobs": -1},
}
MODEL_ORDER: List[str] = ["Logistic Regression", "Decision Tree", "Random Forest"]
