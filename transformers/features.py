"""Transformer block: build the model matrix and the stratified split."""
from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.config import FEATURES, ONEHOT_COLS, TARGET, TEST_SIZE, SEED
from utils.logging_config import get_logger
logger = get_logger(__name__)

def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """One-hot encode zone and inspection type; return (X, y)."""
    X = pd.get_dummies(df[FEATURES], columns=ONEHOT_COLS, drop_first=True)
    return X, df[TARGET]

def split_xy(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    logger.info("Split: train=%d test=%d features=%d", len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test
