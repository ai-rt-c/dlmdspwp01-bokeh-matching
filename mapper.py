from __future__ import annotations

import numpy as np
import pandas as pd

from models import AssignmentRow, MatchResult


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
            raise ValueError("Expected at least 50 ideal series y1..y50.")
        # Ensure x grid matches for inner join by position.
        if not np.array_equal(self._x, ideal_df["x"].to_numpy()):
            # Keep it simple: require exact alignment for matching
            # (classification tolerates only insignificant rounding drift).
            raise ValueError("train and ideal must share the same x grid for matching.")

    def select_best_ideals(self) -> list[MatchResult]:
        results: list[MatchResult] = []
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
            results.append(
                MatchResult(
                    train_series=tcol,
                    ideal_series=best,
                    sse=best_stats[0],
                    mse=best_stats[1],
                    delta_max_abs=best_stats[2],
                )
            )
        return results


def _lookup_ideal_value(
    ideal_df: pd.DataFrame, ideal_col: str, x_val: float, tol: float = 1e-9
) -> tuple[float, str]:
    exact_mask = ideal_df["x"] == x_val
    row = ideal_df.loc[exact_mask]
    if len(row) == 1:
        return float(row[ideal_col].iloc[0]), "exact-x"

    close_mask = np.isclose(
        ideal_df["x"].to_numpy(dtype=float), x_val, atol=tol, rtol=tol
    )
    close_rows = ideal_df.loc[close_mask]
    if len(close_rows) == 1:
        return float(close_rows[ideal_col].iloc[0]), "rounded-x"
    raise ValueError(f"Test x={x_val} is not present on the ideal-function grid.")


def assign_test(
    ideal_df: pd.DataFrame,
    matches: list[MatchResult],
    test_df: pd.DataFrame,
) -> list[AssignmentRow]:
    # Build thresholds per training series
    thresholds = {m.train_series: (m.ideal_series, m.delta_max_abs) for m in matches}
    out: list[AssignmentRow] = []
    for _, row in test_df.iterrows():
        x_t = float(row["x"])
        y_t = float(row["y"])
        candidates = []
        for tcol, (icol, delta) in thresholds.items():
            ideal_y, note = _lookup_ideal_value(ideal_df, icol, x_t)
            resid = y_t - ideal_y
            # Accept if |resid| <= sqrt(2) * delta
            if abs(resid) <= (np.sqrt(2.0) * delta):
                candidates.append((tcol, icol, resid, note))
        if not candidates:
            out.append(
                AssignmentRow(
                    x=x_t,
                    y=y_t,
                    assigned_series=None,
                    ideal_series=None,
                    residual=None,
                    accepted=False,
                    note="no-match",
                )
            )
        else:
            # Tie-break by smallest absolute residual
            tcol, icol, resid, note = min(
                candidates, key=lambda candidate: abs(candidate[2])
            )
            out.append(
                AssignmentRow(
                    x=x_t,
                    y=y_t,
                    assigned_series=tcol,
                    ideal_series=icol,
                    residual=float(resid),
                    accepted=True,
                    note=note,
                )
            )
    return out
