from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

@dataclass
class MatchResult:
    # For a given training series k, store the chosen ideal name and stats.
    train_series: str
    ideal_series: str
    sse: float
    mse: float
    delta_max_abs: float  # max absolute deviation on training grid

@dataclass
class AssignmentRow:
    x: float
    y: float
    assigned_series: str | None
    ideal_series: str | None
    residual: float | None
    accepted: bool
    note: str = ""  # e.g., "exact-x" or "nearest-x"
