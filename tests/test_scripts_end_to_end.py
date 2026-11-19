"""End-to-end runs of the time-series scripts on a small synthetic series.

They replace the manual smoke runs: each script is executed as from the command
line, in a temporary directory, and its output files are checked.
"""

import csv
import json
import sys

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from experiments import data_quality_report, make_synthetic_series  # noqa: E402
from experiments import plot_timeseries_split, train_timeseries  # noqa: E402


def run(monkeypatch, module, *arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
    module.main()


def strict_json(path):
    def reject(token):
        raise ValueError(f"token JSON non standard: {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture
def series(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, make_synthetic_series, "--phi", "0.3", "--count", "400",
        "--output-dir", "series")
    return tmp_path / "series" / "ar1_phi0.3_seed0.csv"


def only(paths):
    paths = list(paths)
    assert len(paths) == 1, paths
    return paths[0]


def test_training_writes_metrics_checkpoint_and_strict_report(series, tmp_path, monkeypatch):
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "2", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1")
    directory = only(tmp_path.glob("outputs/timeseries_*"))

    with (directory / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["epoch"] for row in rows] == ["0", "1", "2"]
    assert rows[0]["train_loss"] == "" and float(rows[2]["train_loss"]) > 0

    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert {"model_state_dict", "epoch", "validation_loss", "calendar", "max_skipped",
            "inputs", "standardizer", "lookback"} <= checkpoint.keys()
    assert checkpoint["validation_loss"] == min(float(row["validation_loss"]) for row in rows)

    report = strict_json(directory / "results.json")
    assert set(report["results"]["test"]) == {"ltc", "zero", "training_mean"}
    assert report["results"]["test"]["zero"]["direction_accuracy"] is None
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("include_test", [False, True])
def test_quality_report_hides_the_test_by_default(series, tmp_path, monkeypatch, include_test):
    arguments = ["--csv", str(series), "--by-decade"] + (["--include-test"] if include_test else [])
    run(monkeypatch, data_quality_report, *arguments)

    report = strict_json(only(tmp_path.glob("outputs/data_quality_*/report.json")))
    expected = {"training", "validation"} | ({"test"} if include_test else set())
    assert set(report["parts"]) == expected
    assert report["decades"]


def test_split_plot_is_saved(series, tmp_path, monkeypatch):
    run(monkeypatch, plot_timeseries_split, "--csv", str(series), "--output", "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
