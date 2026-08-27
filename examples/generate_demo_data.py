"""Generate a deterministic synthetic dataset for the portfolio demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = np.linspace(-10.0, 10.0, 201)

    ideal_values: dict[str, np.ndarray] = {
        "y1": np.sin(x),
        "y2": np.cos(x),
        "y3": 0.1 * x**2,
        "y4": 0.5 * x,
    }
    for index in range(5, 51):
        frequency = 0.15 + (index / 20.0)
        ideal_values[f"y{index}"] = np.sin(frequency * x) + (index * 0.08)

    ideal = pd.DataFrame({"x": x, **ideal_values})
    train = pd.DataFrame(
        {
            "x": x,
            "y1": ideal_values["y1"] + 0.025 * np.cos(2.0 * x),
            "y2": ideal_values["y2"] + 0.020 * np.sin(1.5 * x),
            "y3": ideal_values["y3"] + 0.030 * np.cos(x),
            "y4": ideal_values["y4"] + 0.020 * np.sin(2.5 * x),
        }
    )

    selected = np.arange(0, len(x), 8)
    series = (np.arange(len(selected)) % 4) + 1
    test_y = np.array(
        [
            ideal_values[f"y{series_id}"][row_index]
            for row_index, series_id in zip(selected, series)
        ],
        dtype=float,
    )
    test_y += 0.01 * np.sin(selected)
    test_y[::7] += 2.5
    test = pd.DataFrame({"x": x[selected], "y": test_y})
    return train, ideal, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synthetic CSV inputs for the matching demo."
    )
    parser.add_argument(
        "--output-dir",
        default="examples/data",
        help="Directory for generated CSV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, ideal, test = build_demo_frames()
    train.to_csv(output_dir / "train.csv", index=False)
    ideal.to_csv(output_dir / "ideal.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Demo data written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
