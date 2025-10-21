from __future__ import annotations

import argparse
import pandas as pd
from datasource import load_data, persist_sqlite
from mapper import Matcher, assign_test
from visualize import plot_fit_panels, plot_assignment_scatter

def parse_args():
    p = argparse.ArgumentParser(description="DLMDSPWP01 — Bokeh matching + test assignment")
    p.add_argument("--train", required=True, help="Path to train.csv")
    p.add_argument("--ideal", required=True, help="Path to ideal.csv")
    p.add_argument("--test",  required=True, help="Path to test.csv")
    p.add_argument("--sqlite", default=None, help="Optional: path to SQLite file to persist inputs")
    p.add_argument("--out-html-fit", default=None, help="Optional: Bokeh HTML for train vs ideal")
    p.add_argument("--out-html-cls", default=None, help="Optional: Bokeh HTML for test assignment")
    p.add_argument("--out-csv", default=None, help="Optional: CSV for test mapping results")
    return p.parse_args()

def main():
    args = parse_args()
    dfs = load_data(args.train, args.ideal, args.test)

    if args.sqlite:
        persist_sqlite(dfs, args.sqlite)

    matcher = Matcher(dfs.train, dfs.ideal)
    matches = matcher.select_best_ideals()

    assignments = assign_test(dfs.ideal, matches, dfs.test)

    # Optional CSV export
    if args.out_csv:
        rows = []
        for a in assignments:
            rows.append({
                "x": a.x, "y": a.y,
                "assigned_series": a.assigned_series,
                "ideal_series": a.ideal_series,
                "residual": a.residual,
                "accepted": a.accepted,
                "note": a.note,
            })
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    # Optional visuals
    if args.out_html_fit:
        plot_fit_panels(dfs.train, dfs.ideal, matches, args.out_html_fit)
    if args.out_html_cls:
        plot_assignment_scatter(dfs.test, assignments, args.out_html_cls)

if __name__ == "__main__":
    main()
