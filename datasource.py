from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Simple data access utilities. The goal is clarity and robustness.

@dataclass
class DataFrames:
    train: pd.DataFrame
    ideal: pd.DataFrame
    test: pd.DataFrame

def _try_read_csv(path: str) -> pd.DataFrame:
    # Try common (sep, decimal) pairs, then fallback to pandas defaults.
    trials = [
        { "sep": ",", "decimal": "."},
        { "sep": ";", "decimal": ","},
        { "sep": ";", "decimal": "."},
        { "sep": ",", "decimal": ","},
    ]
    for kw in trials:
        try:
            df = pd.read_csv(path, **kw)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return pd.read_csv(path)

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def _validate_x_monotonic(df: pd.DataFrame, x_col: str = "x") -> None:
    if x_col not in df.columns:
        raise ValueError(f"Missing '{x_col}' column.")
    x = df[x_col].values
    if not (np.all(np.diff(x) > 0)):
        raise ValueError("x must be strictly increasing.")

def load_data(train_path: str, ideal_path: str, test_path: str) -> DataFrames:
    train = _coerce_numeric(_try_read_csv(train_path))
    ideal = _coerce_numeric(_try_read_csv(ideal_path))
    test  = _coerce_numeric(_try_read_csv(test_path))

    required_train = {"x", "y1", "y2", "y3", "y4"}
    if not required_train.issubset(set(train.columns)):
        raise ValueError("train.csv must contain columns: x, y1..y4")
    if "x" not in ideal.columns or len([c for c in ideal.columns if c.startswith("y")]) < 50:
        # Allow flexible naming as long as there are 50 y-columns.
        pass
    if not {"x", "y"}.issubset(set(test.columns)):
        raise ValueError("test.csv must contain columns: x, y")

    _validate_x_monotonic(train, "x")
    _validate_x_monotonic(ideal, "x")
    _validate_x_monotonic(test, "x")

    return DataFrames(train=train, ideal=ideal, test=test)

def persist_sqlite(dfs: DataFrames, sqlite_path: str) -> None:
    engine = create_engine(f"sqlite:///{sqlite_path}")
    dfs.train.to_sql("train", engine, if_exists="replace", index=False)
    dfs.ideal.to_sql("ideal", engine, if_exists="replace", index=False)
    dfs.test.to_sql("test", engine, if_exists="replace", index=False)
