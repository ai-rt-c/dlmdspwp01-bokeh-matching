import numpy as np
import pandas as pd
import pytest

from datasource import load_data


def _write_valid_inputs(tmp_path):
    x = np.arange(5, dtype=float)
    train = pd.DataFrame({"x": x, **{f"y{i}": x + i for i in range(1, 5)}})
    ideal = pd.DataFrame({"x": x, **{f"y{i}": x + i for i in range(1, 51)}})
    test = pd.DataFrame({"x": [2.0, 0.0, 4.0], "y": [3.0, 1.0, 5.0]})
    paths = [tmp_path / name for name in ("train.csv", "ideal.csv", "test.csv")]
    for frame, path in zip((train, ideal, test), paths):
        frame.to_csv(path, index=False)
    return paths


def test_load_data_accepts_unordered_test_points(tmp_path):
    paths = _write_valid_inputs(tmp_path)
    frames = load_data(*(str(path) for path in paths))
    assert frames.train.shape == (5, 5)
    assert frames.ideal.shape == (5, 51)
    assert frames.test["x"].tolist() == [2.0, 0.0, 4.0]


def test_load_data_rejects_incomplete_ideal_schema(tmp_path):
    train_path, ideal_path, test_path = _write_valid_inputs(tmp_path)
    ideal = pd.read_csv(ideal_path).drop(columns="y50")
    ideal.to_csv(ideal_path, index=False)

    with pytest.raises(ValueError, match="y1..y50"):
        load_data(str(train_path), str(ideal_path), str(test_path))


def test_load_data_rejects_non_numeric_values(tmp_path):
    train_path, ideal_path, test_path = _write_valid_inputs(tmp_path)
    test = pd.read_csv(test_path)
    test["y"] = test["y"].astype(object)
    test.loc[0, "y"] = "not-a-number"
    test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="numeric"):
        load_data(str(train_path), str(ideal_path), str(test_path))


def test_load_data_rejects_non_finite_values(tmp_path):
    train_path, ideal_path, test_path = _write_valid_inputs(tmp_path)
    test = pd.read_csv(test_path)
    test.loc[0, "y"] = np.inf
    test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="finite"):
        load_data(str(train_path), str(ideal_path), str(test_path))


def test_load_data_rejects_empty_inputs(tmp_path):
    train_path, ideal_path, test_path = _write_valid_inputs(tmp_path)
    pd.read_csv(test_path).iloc[0:0].to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="at least one row"):
        load_data(str(train_path), str(ideal_path), str(test_path))


def test_load_data_supports_semicolon_csv_with_decimal_commas(tmp_path):
    train_path, ideal_path, test_path = _write_valid_inputs(tmp_path)
    for path in (train_path, ideal_path, test_path):
        frame = pd.read_csv(path)
        frame.to_csv(path, index=False, sep=";", decimal=",")

    frames = load_data(str(train_path), str(ideal_path), str(test_path))
    assert frames.train["y1"].dtype.kind in "fi"
    assert frames.ideal["y50"].dtype.kind in "fi"
