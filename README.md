# Least-Squares Function Matching & Test-Point Assignment

[![Tests](https://github.com/ai-rt-c/dlmdspwp01-bokeh-matching/actions/workflows/tests.yml/badge.svg)](https://github.com/ai-rt-c/dlmdspwp01-bokeh-matching/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Python pipeline that matches four observed training series to 50 candidate functions, derives a data-driven acceptance threshold for each match, and assigns previously unseen test points. Results can be exported to CSV, persisted in SQLite, and explored through interactive Bokeh visualizations.

This project originated in the IU International University module **DLMDSPWP01 — Programming with Python** and has since been upgraded as a maintained portfolio project. The current version adds modern-library compatibility, stricter validation, automated integration tests, and deterministic synthetic demo data. The original course datasets, generated database, and written report are intentionally excluded for redistribution and privacy reasons.

## What the pipeline does

1. Validates and loads training, ideal-function, and test CSV files.
2. Selects the ideal function with the lowest sum of squared errors (SSE) for each training series.
3. Computes the largest absolute training residual for every selected match.
4. Accepts a test point when its deviation is at most `sqrt(2) × max training deviation`.
5. Resolves multiple valid candidates by choosing the smallest absolute residual.
6. Optionally writes the training data, ideal functions, and mapped test results to SQLite and generates interactive Bokeh reports.

## Quick start

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python examples/generate_demo_data.py
```

Run the full demo:

```bash
python main.py \
  --train examples/data/train.csv \
  --ideal examples/data/ideal.csv \
  --test examples/data/test.csv \
  --sqlite artifacts/demo.sqlite \
  --out-csv artifacts/test_mapping_results.csv \
  --out-html-fit artifacts/train_vs_ideal.html \
  --out-html-cls artifacts/test_classification.html
```

The command creates two standalone interactive HTML reports, the classified test-point CSV, and an optional SQLite database in `artifacts/`. The database contains `train_data`, `ideal_functions`, and `test_results`; the result table records the mapped training series and absolute deviation for each test point. Generated data and artifacts are ignored by Git.

## Input contracts

| File | Required columns | Additional rule |
| --- | --- | --- |
| `train.csv` | `x, y1, y2, y3, y4` | `x` must be strictly increasing |
| `ideal.csv` | `x, y1 ... y50` | Must use the same `x` grid as training |
| `test.csv` | `x, y` | Rows may be in any order |

Comma- and semicolon-separated CSV files are supported, including comma decimals. All values must be numeric and non-missing.

## Tests

```bash
python -m pytest -q
```

The test suite covers ideal-function selection, exact threshold boundaries, out-of-grid rejection, CSV validation, unordered test points, schema and finite-value failures, SQLite result persistence, and Bokeh artifact generation. GitHub Actions runs it on Python 3.11 and 3.12 for every push and pull request.

## Repository structure

```text
main.py                    Command-line entry point
datasource.py              CSV validation and optional SQLite persistence
mapper.py                  SSE matching and threshold-based assignment
models.py                  Typed result records
visualize.py               Interactive Bokeh output
examples/generate_demo_data.py
                            Reproducible synthetic inputs
tests/                     Automated tests
submission_single.py       Single-file course-submission variant
```

## Scope and limitations

- Matching assumes that training and ideal functions share the same `x` grid.
- Test points must be on the ideal-function grid, allowing only insignificant floating-point rounding differences.
- The method is deterministic and intentionally focuses on the assignment's least-squares rule; it is not a probabilistic forecasting model.

## License

MIT — see [LICENSE](LICENSE).
