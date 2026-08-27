from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from models import AssignmentRow

# Simple data access utilities. The goal is clarity and robustness.


@dataclass
class DataFrames:
    train: pd.DataFrame
    ideal: pd.DataFrame
    test: pd.DataFrame


def _try_read_csv(path: str) -> pd.DataFrame:
    sample = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:4096]
    header = sample.splitlines()[0] if sample else ""
    separator = ";" if header.count(";") > header.count(",") else ","
    decimal = "," if separator == ";" and re.search(r"\d,\d", sample) else "."
    df = pd.read_csv(path, sep=separator, decimal=decimal)
    if df.shape[1] < 2:
        raise ValueError(f"Could not detect at least two CSV columns in '{path}'.")
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        try:
            out[c] = pd.to_numeric(out[c], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{c}' must contain only numeric values.") from exc
    if out.isna().any().any():
        raise ValueError("Input data must not contain missing numeric values.")
    if out.empty:
        raise ValueError("Input data must contain at least one row.")
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("Input data must contain only finite numeric values.")
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
    test = _coerce_numeric(_try_read_csv(test_path))

    required_train = {"x", "y1", "y2", "y3", "y4"}
    if not required_train.issubset(set(train.columns)):
        raise ValueError("train.csv must contain columns: x, y1..y4")
    required_ideal = {"x", *(f"y{i}" for i in range(1, 51))}
    if not required_ideal.issubset(set(ideal.columns)):
        raise ValueError("ideal.csv must contain columns: x, y1..y50")
    if not {"x", "y"}.issubset(set(test.columns)):
        raise ValueError("test.csv must contain columns: x, y")

    _validate_x_monotonic(train, "x")
    _validate_x_monotonic(ideal, "x")

    return DataFrames(train=train, ideal=ideal, test=test)


def persist_sqlite(
    dfs: DataFrames,
    assignments: list[AssignmentRow],
    sqlite_path: str,
) -> None:
    """Persist the assignment's required input and result tables.

    The frozen course submission uses ``train_data``, ``ideal_functions``, and
    ``test_results``. Keeping those names on the maintained branch makes the
    generated database easy to compare with the original deliverable.
    """
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    test_results = pd.DataFrame(
        [
            {
                "x": assignment.x,
                "y": assignment.y,
                "mapped_train": assignment.assigned_series or "No Match",
                "deviation": (
                    abs(assignment.residual)
                    if assignment.residual is not None
                    else None
                ),
            }
            for assignment in assignments
        ],
        columns=["x", "y", "mapped_train", "deviation"],
    )

    engine = create_engine(URL.create("sqlite", database=sqlite_path))
    try:
        with engine.begin() as connection:
            dfs.train.to_sql("train_data", connection, if_exists="replace", index=False)
            dfs.ideal.to_sql(
                "ideal_functions", connection, if_exists="replace", index=False
            )
            test_results.to_sql(
                "test_results", connection, if_exists="replace", index=False
            )
    finally:
        engine.dispose()
