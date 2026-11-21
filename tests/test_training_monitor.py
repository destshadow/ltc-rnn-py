"""Live training metrics: the log written by training and read by the monitor."""

import math

import pytest

from experiments.watch_training import latest_metrics, read_metrics
from training.metrics_log import MetricsLog


def test_log_round_trip_and_epoch_zero_without_training(tmp_path):
    path = tmp_path / "timeseries_run" / "metrics.csv"
    path.parent.mkdir()
    log = MetricsLog(path, validation_baseline=1.25)
    log.write(0, None, 1.3)
    log.write(1, 0.9, 1.2)

    metrics = read_metrics(path)
    assert metrics["epoch"] == [0.0, 1.0]
    assert math.isnan(metrics["train_loss"][0]) and metrics["train_loss"][1] == 0.9
    assert metrics["validation_loss"] == [1.3, 1.2]
    assert metrics["validation_baseline"] == [1.25, 1.25]
    assert metrics["seconds"][1] >= metrics["seconds"][0] >= 0

    assert latest_metrics(tmp_path) == path
    with pytest.raises(FileExistsError):  # a run never overwrites its own log
        MetricsLog(path, validation_baseline=1.0)


def test_missing_runs_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_metrics(tmp_path)


def test_monitor_waits_for_the_first_row_then_draws(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.watch_training import draw

    path = tmp_path / "metrics.csv"
    log = MetricsLog(path, validation_baseline=1.0)   # header only, as right after start
    figure, axis = plt.subplots()

    draw(axis, read_metrics(path), "run")             # must not fail on an empty log
    assert "attesa" in axis.get_title()

    log.write(0, None, 1.2)                           # first row arrives, same figure
    draw(axis, read_metrics(path), "run")
    assert "epoca 0" in axis.get_title()
    plt.close(figure)


def test_once_on_an_empty_log_saves_a_waiting_plot(tmp_path, monkeypatch):
    import sys

    from experiments import watch_training

    path = tmp_path / "metrics.csv"
    MetricsLog(path, validation_baseline=1.0)
    monkeypatch.setattr(sys, "argv", ["watch_training", "--metrics", str(path), "--once"])
    watch_training.main()

    assert (tmp_path / "loss.png").stat().st_size > 0


def test_latest_metrics_also_finds_volatility_runs(tmp_path):
    import os

    older = tmp_path / "timeseries_a" / "metrics.csv"
    newer = tmp_path / "volatility_b" / "metrics.csv"
    for path in (older, newer):
        path.parent.mkdir()
        MetricsLog(path, validation_baseline=1.0)
    os.utime(older, (1, 1))   # make the returns run clearly older

    assert latest_metrics(tmp_path) == newer


def test_latest_metrics_also_finds_event_model_runs(tmp_path):
    import os

    older = tmp_path / "volatility_a" / "metrics.csv"
    newer = tmp_path / "event_model_b" / "metrics.csv"
    for path in (older, newer):
        path.parent.mkdir()
        MetricsLog(path, validation_baseline=1.0)
    os.utime(older, (1, 1))

    assert latest_metrics(tmp_path) == newer
