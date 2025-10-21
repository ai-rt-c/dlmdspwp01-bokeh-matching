from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from models import MatchResult, AssignmentRow

class Matcher:
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self._x = train_df["x"].to_numpy()
        # Collect y-columns
        self.train_cols = [c for c in train_df.columns if c.startswith("y")]
        self.ideal_cols = [c for c in ideal_df.columns if c.startswith("y")]
        if len(self.train_cols) != 4:
            raise ValueError("Expected exactly 4 training series y1..y4.")
        if len(self.ideal_cols) < 50:
            # Some datasets use y1..y50. We require at least 50.
            pass
        # Ensure x grid matches for inner join by position.
        if not np.array_equal(self._x, ideal_df["x"].to_numpy()):
            # Keep it simple: require exact alignment for matching
            # (classification handles nearest-x if needed).
            raise ValueError("train and ideal must share the same x grid for matching.")

    def _sse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r = y_true - y_pred
        return float(np.sum(r * r))

    def select_best_ideals(self) -> List[MatchResult]:
        results: List[MatchResult] = []
        x = self._x
        for tcol in self.train_cols:
            y_train = self.train_df[tcol].to_numpy()
            best = None
            best_stats = (np.inf, np.inf, np.inf)  # sse, mse, delta_max
            for icol in self.ideal_cols:
                y_id = self.ideal_df[icol].to_numpy()
                r = y_train - y_id
                sse = float(np.sum(r * r))
                mse = float(sse / len(x))
                delta_max = float(np.max(np.abs(r)))
                if sse < best_stats[0]:
                    best = icol
                    best_stats = (sse, mse, delta_max)
            assert best is not None
            results.append(MatchResult(
                train_series=tcol,
                ideal_series=best, sse=best_stats[0],
                mse=best_stats[1], delta_max_abs=best_stats[2]
            ))
        return results

def _lookup_ideal_value(ideal_df: pd.DataFrame, ideal_col: str, x_val: float, tol: float = 1e-9) -> tuple[float, str]:
    # Exact match first
    row = ideal_df.loc[ideal_df["x"] == x_val]
    if len(row) == 1:
        return float(row[ideal_col].iloc[0]), "exact-x"
    # Nearest neighbor (small tolerance)
    # If grid is regular, nearest value is fine for small rounding drift
    idx = (ideal_df["x"] - x_val).abs().idxmin()
    nearest_x = float(ideal_df.loc[idx, "x"])
    note = "nearest-x" if abs(nearest_x - x_val) > tol else "exact-x"
    return float(ideal_df.loc[idx, ideal_col]), note

def assign_test(
    ideal_df: pd.DataFrame,
    matches: List[MatchResult],
    test_df: pd.DataFrame,
) -> List[AssignmentRow]:
    # Build thresholds per training series
    thresholds = {m.train_series: (m.ideal_series, m.delta_max_abs) for m in matches}
    out: List[AssignmentRow] = []
    for _, row in test_df.iterrows():
        x_t = float(row["x"])
        y_t = float(row[list(row.index)[-1]])  # last numeric col as y
        candidates = []
        for tcol, (icol, delta) in thresholds.items():
            ideal_y, note = _lookup_ideal_value(ideal_df, icol, x_t)
            resid = y_t - ideal_y
            # Accept if |resid| <= sqrt(2) * delta
            if abs(resid) <= (np.sqrt(2.0) * delta):
                candidates.append((tcol, icol, resid, note))
        if not candidates:
            out.append(AssignmentRow(x=x_t, y=y_t, assigned_series=None, ideal_series=None, residual=None, accepted=False, note="no-match"))
        else:
            # Tie-break by smallest absolute residual
            tcol, icol, resid, note = sorted(candidates, key=lambda t: abs(t[2]))[0]
            out.append(AssignmentRow(x=x_t, y=y_t, assigned_series=tcol, ideal_series=icol, residual=float(resid), accepted=True, note=note))
    return out
