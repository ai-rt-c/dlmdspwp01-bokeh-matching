from __future__ import annotations

from typing import List
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Legend

# Minimal, readable Bokeh views used for grading by inspection.

def plot_fit_panels(train_df: pd.DataFrame, ideal_df: pd.DataFrame, matches, out_html: str) -> None:
    figs = []
    for m in matches:
        tcol, icol = m.train_series, m.ideal_series
        src = ColumnDataSource({
            "x": train_df["x"],
            "y_train": train_df[tcol],
            "y_ideal": ideal_df[icol],
            "resid": train_df[tcol] - ideal_df[icol],
        })
        p = figure(title=f"Train vs Ideal — {tcol} ↔ {icol}", width=900, height=280)
        r1 = p.line("x", "y_train", source=src, line_width=2)
        r2 = p.line("x", "y_ideal", source=src, line_width=2, line_dash="dashed")
        p.add_tools(HoverTool(tooltips=[("x","@x"), ("train","@y_train"), ("ideal","@y_ideal"), ("resid","@resid")]))
        legend = Legend(items=[(f"{tcol}", [r1]), (f"{icol}", [r2])])
        p.add_layout(legend, "right")
        p.legend.click_policy = "hide"
        figs.append(p)
    output_file(out_html, title="Train vs Ideal")
    save(column(*figs))

def plot_assignment_scatter(test_df: pd.DataFrame, assignments, out_html: str) -> None:
    # Prepare tidy frame
    rows = []
    for a in assignments:
        rows.append({
            "x": a.x, "y": a.y,
            "assigned": "yes" if a.accepted else "no",
            "series": a.assigned_series if a.assigned_series else "None",
            "note": a.note
        })
    df = pd.DataFrame(rows)
    src_yes = ColumnDataSource(df[df["assigned"]=="yes"])
    src_no  = ColumnDataSource(df[df["assigned"]=="no"])
    p = figure(title="Test Assignment", width=900, height=400)
    r1 = p.circle(x="x", y="y", size=6, alpha=0.9, source=src_yes, legend_label="accepted")
    r2 = p.circle_x(x="x", y="y", size=7, alpha=0.9, source=src_no, legend_label="rejected")
    p.add_tools(HoverTool(tooltips=[("x","@x"), ("y","@y"), ("assigned","@assigned"), ("series","@series"), ("note","@note")]))
    p.legend.click_policy = "hide"
    output_file(out_html, title="Test Assignment")
    save(p)
