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


def test_decade_runs_never_cross_a_gap_or_a_decade():
    from datetime import datetime, timedelta

    from timeseries.dataset import Piece

    day = timedelta(days=1)
    first = Piece(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3),
                  [datetime(1959, 12, 30) + k * day for k in range(3)])   # 30/12/1959 → 01/01/1960
    second = Piece(torch.tensor([4.0, 5.0]), torch.ones(2),
                   [datetime(1960, 1, 10) + k * day for k in range(2)])   # after a gap

    runs = data_quality_report.decade_runs([first, second])
    assert [run.tolist() for run in runs[1950]] == [[1.0, 2.0]]
    assert [run.tolist() for run in runs[1960]] == [[3.0], [4.0, 5.0]]


def test_partial_or_constant_groups_do_not_stop_the_report(tmp_path, monkeypatch):
    from datetime import date, timedelta

    monkeypatch.chdir(tmp_path)
    # Starts on 2009-12-30: the 2000s get a single return, the 2010s the rest.
    rows = ["Date,Close"] + [f"{date(2009, 12, 30) + timedelta(days=k)},{100 + (k % 7)}"
                             for k in range(300)]
    (tmp_path / "prices.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    run(monkeypatch, data_quality_report, "--csv", "prices.csv", "--by-decade", "--include-test")

    report = strict_json(only(tmp_path.glob("outputs/data_quality_*/report.json")))
    assert report["decades"]["2000"]["count"] == 1
    assert "unavailable" in report["decades"]["2000"]
    assert "lag1_autocorrelation" in report["decades"]["2010"]
    assert {"training", "validation", "test"} == set(report["parts"])

    constant = data_quality_report.safe_stats([torch.ones(10)])
    assert constant["count"] == 10 and "costanti" in constant["unavailable"]


def test_training_with_date_split_records_the_dates(series, tmp_path, monkeypatch):
    run(monkeypatch, train_timeseries, "--csv", str(series), "--epochs", "1", "--lookback", "5",
        "--hidden-size", "2", "--substeps", "1", "--train-start", "2000-03-01",
        "--validation-start", "2000-09-01", "--test-start", "2000-11-01")

    report = strict_json(only(tmp_path.glob("outputs/timeseries_*/results.json")))
    arguments = report["protocol"]["arguments"]
    assert arguments["train_start"].startswith("2000-03-01")
    assert arguments["test_start"].startswith("2000-11-01")
    assert report["results"]["test"]["ltc"]["count"] > 0
