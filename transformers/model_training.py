"""
Transformer block: train and evaluate the three classifiers.

Cross-validates ROC-AUC on the training split, fits on train, scores on test,
and reports Random-Forest feature importance with the share attributable to
zone features (the project's central question).
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from utils.config import MODEL_PARAMS, MODEL_ORDER, SEED
from utils.logging_config import get_logger
logger = get_logger(__name__)

def _estimator(name: str, seed: int):
    p = MODEL_PARAMS[name]
    if name == "Logistic Regression": return LogisticRegression(random_state=seed, **p)
    if name == "Decision Tree": return DecisionTreeClassifier(random_state=seed, **p)
    if name == "Random Forest": return RandomForestClassifier(random_state=seed, **p)
    raise ValueError(name)

def train_and_evaluate(X_train, X_test, y_train, y_test, seed: int = SEED
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return (metrics_table, rf_importance, extras with zone-importance share)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name in MODEL_ORDER:
        model = _estimator(name, seed)
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        pr, rc, _ = precision_recall_curve(y_test, proba)
        rows.append({"Model": name,
                     "CV_ROC_AUC": round(cv_auc.mean(), 4),
                     "Accuracy": round(accuracy_score(y_test, pred), 4),
                     "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
                     "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
                     "F1": round(f1_score(y_test, pred, zero_division=0), 4),
                     "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
                     "PR_AUC": round(auc(rc, pr), 4)})
        fitted[name] = model
        logger.info("%-20s CV-AUC %.4f  test ROC-AUC %.4f",
                    name, rows[-1]["CV_ROC_AUC"], rows[-1]["ROC_AUC"])

    rf = fitted["Random Forest"]
    fi = (pd.DataFrame({"feature": X_train.columns, "importance": rf.feature_importances_})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    zone_share = float(fi[fi["feature"].str.startswith("zone_")]["importance"].sum())
    extras = {"rf_zone_importance_share": round(zone_share, 4),
              "rf_type_year_importance_share": round(1 - zone_share, 4)}
    logger.info("RF importance to zone: %.2f%% vs type+year %.2f%%",
                zone_share * 100, (1 - zone_share) * 100)
    return pd.DataFrame(rows), fi, extras
