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
