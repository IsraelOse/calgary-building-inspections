"""Structured logging shared by every block."""
from __future__ import annotations
import logging, sys
def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s","%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
