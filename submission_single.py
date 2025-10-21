
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column
    from bokeh.models import ColumnDataSource, HoverTool, Legend
    _HAS_BOKEH = True
except Exception:
    _HAS_BOKEH = False

@dataclass
class DataFrames:
    train: pd.DataFrame
    ideal: pd.DataFrame
    test: pd.DataFrame

def _try_read_csv(path: str) -> pd.DataFrame:
    trials = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": ","},
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

def _validate_x(df: pd.DataFrame):
    if "x" not in df.columns:
        raise ValueError("Missing 'x' column.")
    x = df["x"].to_numpy()
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

def load_data(train_path: str, ideal_path: str, test_path: str) -> DataFrames:
    train = _coerce_numeric(_try_read_csv(train_path))
    ideal = _coerce_numeric(_try_read_csv(ideal_path))
    test  = _coerce_numeric(_try_read_csv(test_path))

    if not {"x", "y1", "y2", "y3", "y4"}.issubset(set(train.columns)):
        raise ValueError("train.csv must contain columns: x, y1..y4")
    if "x" not in ideal.columns or len([c for c in ideal.columns if c.startswith("y")]) < 50:
        pass
    if not {"x", "y"}.issubset(set(test.columns)):
        raise ValueError("test.csv must contain columns: x, y")

    _validate_x(train); _validate_x(ideal); _validate_x(test)
    return DataFrames(train=train, ideal=ideal, test=test)

@dataclass
class MatchResult:
    train_series: str
    ideal_series: str
    sse: float
    mse: float
    delta_max_abs: float

@dataclass
class AssignmentRow:
    x: float
    y: float
    assigned_series: str | None
    ideal_series: str | None
    residual: float | None
    accepted: bool
    note: str = ""

class Matcher:
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.x = train_df["x"].to_numpy()
        self.train_cols = [c for c in train_df.columns if c.startswith("y")]
        self.ideal_cols = [c for c in ideal_df.columns if c.startswith("y")]
        if len(self.train_cols) != 4:
            raise ValueError("Expected 4 training series y1..y4.")
        if not np.array_equal(self.x, ideal_df["x"].to_numpy()):
            raise ValueError("train and ideal must share the same x grid (for matching).")

    def select_best_ideals(self) -> List[MatchResult]:
        results: List[MatchResult] = []
        for tcol in self.train_cols:
            y_t = self.train_df[tcol].to_numpy()
            best = None
            best_stats = (np.inf, np.inf, np.inf)
            for icol in self.ideal_cols:
                y_i = self.ideal_df[icol].to_numpy()
                r = y_t - y_i
                sse = float(np.sum(r*r))
                mse = float(sse / len(self.x))
                delta_max = float(np.max(np.abs(r)))
                if sse < best_stats[0]:
                    best = icol
                    best_stats = (sse, mse, delta_max)
            results.append(MatchResult(tcol, best, best_stats[0], best_stats[1], best_stats[2]))
        return results

def _lookup_ideal_value(ideal_df: pd.DataFrame, ideal_col: str, x_val: float, tol: float = 1e-9) -> tuple[float, str]:
    row = ideal_df.loc[ideal_df["x"] == x_val]
    if len(row) == 1:
        return float(row[ideal_col].iloc[0]), "exact-x"
    idx = (ideal_df["x"] - x_val).abs().idxmin()
    x_near = float(ideal_df.loc[idx, "x"])
    note = "nearest-x" if abs(x_near - x_val) > tol else "exact-x"
    return float(ideal_df.loc[idx, ideal_col]), note

def assign_test(ideal_df: pd.DataFrame, matches: List[MatchResult], test_df: pd.DataFrame) -> List[AssignmentRow]:
    thresholds = {m.train_series: (m.ideal_series, m.delta_max_abs) for m in matches}
    out: List[AssignmentRow] = []
    for _, row in test_df.iterrows():
        x_t = float(row["x"])
        y_t = float(row[list(row.index)[-1]])
        candidates = []
        for tcol, (icol, delta) in thresholds.items():
            val, note = _lookup_ideal_value(ideal_df, icol, x_t)
            resid = y_t - val
            if abs(resid) <= (np.sqrt(2.0) * delta):
                candidates.append((tcol, icol, resid, note))
        if not candidates:
            out.append(AssignmentRow(x_t, y_t, None, None, None, False, "no-match"))
        else:
            tcol, icol, resid, note = sorted(candidates, key=lambda t: abs(t[2]))[0]
            out.append(AssignmentRow(x_t, y_t, tcol, icol, float(resid), True, note))
    return out

def plot_fit_panels(train_df: pd.DataFrame, ideal_df: pd.DataFrame, matches: List[MatchResult], out_html: str) -> None:
    if not _HAS_BOKEH:
        return
    figs = []
    for m in matches:
        tcol, icol = m.train_series, m.ideal_series
        src = ColumnDataSource({"x": train_df["x"], "y_train": train_df[tcol], "y_ideal": ideal_df[icol],
                                "resid": train_df[tcol] - ideal_df[icol]})
        p = figure(title=f"Train vs Ideal — {tcol} ↔ {icol}", width=900, height=280)
        r1 = p.line("x", "y_train", source=src, line_width=2)
        r2 = p.line("x", "y_ideal", source=src, line_width=2, line_dash="dashed")
        p.add_tools(HoverTool(tooltips=[("x","@x"), ("train","@y_train"), ("ideal","@y_ideal"), ("resid","@resid")]))
        legend = Legend(items=[(tcol, [r1]), (icol, [r2])])
        p.add_layout(legend, "right"); p.legend.click_policy = "hide"
        figs.append(p)
    output_file(out_html, title="Train vs Ideal"); save(column(*figs))

def plot_assignment_scatter(test_df: pd.DataFrame, assignments: List[AssignmentRow], out_html: str) -> None:
    if not _HAS_BOKEH:
        return
    rows = []
    for a in assignments:
        rows.append({"x": a.x, "y": a.y, "assigned": "yes" if a.accepted else "no",
                     "series": a.assigned_series if a.assigned_series else "None",
                     "note": a.note})
    df = pd.DataFrame(rows)
    src_yes = ColumnDataSource(df[df["assigned"]=="yes"])
    src_no  = ColumnDataSource(df[df["assigned"]=="no"])
    p = figure(title="Test Assignment", width=900, height=400)
    p.circle(x="x", y="y", size=6, alpha=0.9, source=src_yes, legend_label="accepted")
    p.circle_x(x="x", y="y", size=7, alpha=0.9, source=src_no, legend_label="rejected")
    p.add_tools(HoverTool(tooltips=[("x","@x"), ("y","@y"), ("assigned","@assigned"), ("series","@series"), ("note","@note")]))
    p.legend.click_policy = "hide"
    output_file(out_html, title="Test Assignment"); save(p)

def parse_args():
    p = argparse.ArgumentParser(description="DLMDSPWP01 — single-file matching + test assignment")
    p.add_argument("--train", required=True)
    p.add_argument("--ideal", required=True)
    p.add_argument("--test",  required=True)
    p.add_argument("--out-html-fit", default=None)
    p.add_argument("--out-html-cls", default=None)
    p.add_argument("--out-csv", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    dfs = load_data(args.train, args.ideal, args.test)
    matches = Matcher(dfs.train, dfs.ideal).select_best_ideals()
    assignments = assign_test(dfs.ideal, matches, dfs.test)

    if args.out_csv:
        rows = [{"x": a.x, "y": a.y, "assigned_series": a.assigned_series, "ideal_series": a.ideal_series,
                 "residual": a.residual, "accepted": a.accepted, "note": a.note} for a in assignments]
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    if args.out_html_fit:
        plot_fit_panels(dfs.train, dfs.ideal, matches, args.out_html_fit)
    if args.out_html_cls:
        plot_assignment_scatter(dfs.test, assignments, args.out_html_cls)

if __name__ == "__main__":
    main()
