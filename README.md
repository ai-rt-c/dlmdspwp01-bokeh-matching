# DLMDSPWP01 — Bokeh-based Matching & Test Assignment

This repository contains a small pipeline that selects one ideal function for each training series (least squares)
and classifies test points with a series-specific threshold derived from the training residuals.
Outputs can be rendered as Bokeh HTML files.

## Run (local)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python main.py   --train path/to/train.csv   --ideal path/to/ideal.csv   --test path/to/test.csv   --out-html-fit bokeh_train_vs_ideal.html   --out-html-cls bokeh_test_classification.html   --out-csv test_mapping_results.csv
```
Notes:
- Inputs are not included.
- CSV schemas:
  - train.csv: columns `x, y1, y2, y3, y4`
  - ideal.csv: columns `x, y1..y50`
  - test.csv: columns `x, y`
- Artifacts (HTML/CSV/DB) should not be committed to the repo.
