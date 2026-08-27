import sqlite3
import sys

import main as app
from examples.generate_demo_data import build_demo_frames


def test_cli_creates_all_requested_artifacts(tmp_path, monkeypatch):
    train, ideal, test = build_demo_frames()
    data_dir = tmp_path / "data"
    train_path = data_dir / "train.csv"
    ideal_path = data_dir / "ideal.csv"
    test_path = data_dir / "test.csv"
    data_dir.mkdir()
    train.to_csv(train_path, index=False)
    ideal.to_csv(ideal_path, index=False)
    test.to_csv(test_path, index=False)

    artifact_dir = tmp_path / "nested" / "artifacts"
    sqlite_path = artifact_dir / "demo.sqlite"
    csv_path = artifact_dir / "assignments.csv"
    fit_path = artifact_dir / "fit.html"
    classification_path = artifact_dir / "classification.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--train",
            str(train_path),
            "--ideal",
            str(ideal_path),
            "--test",
            str(test_path),
            "--sqlite",
            str(sqlite_path),
            "--out-csv",
            str(csv_path),
            "--out-html-fit",
            str(fit_path),
            "--out-html-cls",
            str(classification_path),
        ],
    )

    app.main()

    for path in (sqlite_path, csv_path, fit_path, classification_path):
        assert path.is_file()
        assert path.stat().st_size > 0

    with sqlite3.connect(sqlite_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        result_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(test_results)")
        ]
        result_count = connection.execute(
            "SELECT COUNT(*) FROM test_results"
        ).fetchone()[0]
        train_count = connection.execute("SELECT COUNT(*) FROM train_data").fetchone()[
            0
        ]
        ideal_count = connection.execute(
            "SELECT COUNT(*) FROM ideal_functions"
        ).fetchone()[0]
        invalid_deviation_count = connection.execute(
            """SELECT COUNT(*) FROM test_results
               WHERE deviation < 0
                  OR (mapped_train = 'No Match' AND deviation IS NOT NULL)"""
        ).fetchone()[0]

    assert tables == {"train_data", "ideal_functions", "test_results"}
    assert result_columns == ["x", "y", "mapped_train", "deviation"]
    assert result_count == len(test)
    assert train_count == len(train)
    assert ideal_count == len(ideal)
    assert invalid_deviation_count == 0
