"""Time-series data handling: alignment, chronological order and no leakage."""

import math
from datetime import datetime, timedelta

import pytest
import torch

from ltc.config import LTCConfig
from timeseries.calendar import elapsed_periods, expected_periods, split_segments
from timeseries.evaluation import baseline_predictions, predict, regression_report
from timeseries.model import ReturnForecaster
from timeseries.prices import iter_arff_prices, iter_csv_prices, resample_last
from timeseries.windows import (
    Standardizer,
    chronological_split,
    log_returns,
    make_windows,
    walk_forward_folds,
)


def test_log_returns():
    prices = torch.tensor([100.0, 110.0, 99.0], dtype=torch.float64)
    expected = torch.tensor([math.log(1.1), math.log(0.9)], dtype=torch.float64)
    torch.testing.assert_close(log_returns(prices), expected)

    for invalid in ([1.0], [1.0, 0.0], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            log_returns(torch.tensor(invalid))


def test_windows_target_is_strictly_after_inputs():
    series = torch.arange(10.0)
    inputs, targets = make_windows(series, lookback=3)

    assert inputs.shape == (7, 3, 1)
    for index in range(7):
        torch.testing.assert_close(inputs[index, :, 0], series[index:index + 3])
        assert targets[index] == series[index + 3]

    with pytest.raises(ValueError):
        make_windows(series, lookback=10)


def test_chronological_split_is_contiguous_and_ordered():
    series = torch.arange(100.0)
    training, validation, test = chronological_split(
        series, validation_fraction=0.15, test_fraction=0.15
    )

    assert (len(training), len(validation), len(test)) == (70, 15, 15)
    torch.testing.assert_close(torch.cat([training, validation, test]), series)
    assert training.max() < validation.min() and validation.max() < test.min()

    with pytest.raises(ValueError):
        chronological_split(series, validation_fraction=0.5, test_fraction=0.5)


def test_standardizer_ignores_future_values():
    training = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    scaler = Standardizer.fit(training)
    assert (scaler.mean, scaler.std) == pytest.approx((2.0, 1.0))

    # Transforming later data uses the training statistics unchanged.
    future = torch.tensor([1000.0], dtype=torch.float64)
    torch.testing.assert_close(scaler.transform(future), future - 2.0)
    torch.testing.assert_close(scaler.inverse(scaler.transform(future)), future)

    with pytest.raises(ValueError):
        Standardizer.fit(torch.ones(5))


def test_walk_forward_folds_only_move_forward():
    folds = walk_forward_folds(10, initial=4, step=3)
    assert folds == [(range(0, 4), range(4, 7)), (range(0, 7), range(7, 10))]
    for training, test in folds:
        assert max(training) < min(test)


def test_resample_keeps_last_price_of_each_period():
    start = datetime(2024, 1, 1, 22)
    samples = [
        (start, 1.0),
        (start + timedelta(hours=1), 2.0),
        (start + timedelta(hours=3), 3.0),   # next day
        (start + timedelta(days=2), 4.0),    # third consecutive day
    ]
    starts, prices = resample_last(samples, period=timedelta(days=1))

    assert starts == [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    torch.testing.assert_close(
        prices, torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="crescenti"):
        resample_last(list(reversed(samples)), period=timedelta(days=1))


@pytest.mark.parametrize("period", [timedelta(hours=1), timedelta(days=1)])
def test_missing_period_stays_visible_and_splits_the_series(period):
    start = datetime(2024, 1, 1)
    samples = [(start, 10.0), (start + 2 * period, 12.0), (start + 3 * period, 13.0)]
    starts, prices = resample_last(samples, period=period)

    # The gap is not compressed: the returned starts keep it.
    assert starts == [start, start + 2 * period, start + 3 * period]
    torch.testing.assert_close(elapsed_periods(starts, period=period),
                               torch.tensor([2.0, 1.0], dtype=torch.float64))

    segments = split_segments(starts, prices, period=period, calendar="continuous")
    assert [segment_starts for segment_starts, _ in segments] == [
        [start], [start + 2 * period, start + 3 * period],
    ]

    tolerant = split_segments(
        starts, prices, period=period, calendar="continuous", max_skipped=1
    )
    assert len(tolerant) == 1


def test_weekdays_calendar_separates_closures_from_missing_days():
    day = timedelta(days=1)
    friday = datetime(2024, 1, 5)
    starts = [friday - day, friday, friday + 3 * day, friday + 5 * day]  # Thu Fri Mon Wed
    prices = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    # The weekend is expected; Tuesday is missing data.
    assert expected_periods(friday, friday + 3 * day, period=day, calendar="weekdays") == 1
    segments = split_segments(starts, prices, period=day, calendar="weekdays")
    assert [segment_starts for segment_starts, _ in segments] == [starts[:3], starts[3:]]

    # A holiday can be tolerated explicitly.
    assert len(split_segments(
        starts, prices, period=day, calendar="weekdays", max_skipped=1
    )) == 1

    # Calendar time still distinguishes Fri -> Mon from Thu -> Fri.
    torch.testing.assert_close(elapsed_periods(starts, period=day),
                               torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64))

    assert expected_periods(
        datetime(2024, 1, 1), datetime(2024, 1, 15), period=day, calendar="weekdays"
    ) == 10


def test_calendar_rejects_inconsistent_inputs():
    day = timedelta(days=1)
    saturday = datetime(2024, 1, 6)
    prices = torch.tensor([1.0, 2.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="chiusura"):
        split_segments([saturday - day, saturday], prices, period=day, calendar="weekdays")
    with pytest.raises(ValueError, match="un giorno"):
        split_segments([saturday], prices[:1], period=timedelta(hours=1), calendar="weekdays")
    with pytest.raises(ValueError, match="non supportato"):
        split_segments([saturday], prices[:1], period=day, calendar="lunar")


def test_read_arff_skips_missing_values(tmp_path):
    path = tmp_path / "prices.arff"
    path.write_text(
        "% comment\n@RELATION BTC\n\n"
        "@ATTRIBUTE Timestamp STRING\n@ATTRIBUTE Open REAL\n@ATTRIBUTE Close REAL\n"
        "@DATA\n"
        "'2024-01-01 00:00:00',1.0,1.5\n"
        "'2024-01-01 00:01:00',?,?\n"
        "'2024-01-01 00:02:00',2.0,2.5\n",
        encoding="utf-8",
    )
    assert list(iter_arff_prices(path)) == [
        (datetime(2024, 1, 1, 0, 0), 1.5),
        (datetime(2024, 1, 1, 0, 2), 2.5),
    ]

    with pytest.raises(ValueError, match="Volume"):
        list(iter_arff_prices(path, price_attribute="Volume"))


def test_read_csv(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "Date,Close\n2024-01-01,10\n2024-01-02,\n2024-01-03,12.5\n",
        encoding="utf-8",
    )
    assert list(iter_csv_prices(path)) == [
        (datetime(2024, 1, 1), 10.0),
        (datetime(2024, 1, 3), 12.5),
    ]
    # A row with a missing price must not silently shorten the elapsed time.
    with pytest.raises(ValueError, match="Periodi mancanti"):
        resample_last(iter_csv_prices(path), period=timedelta(days=1))


def test_forecaster_shapes_and_gradients():
    torch.manual_seed(0)
    model = ReturnForecaster(LTCConfig(1, 4))
    inputs = torch.randn(5, 12, 1)

    output = model(inputs, dt=0.1)
    assert output.shape == (5,)
    output.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    torch.testing.assert_close(predict(model, inputs, dt=0.1, batch_size=2), output.detach())


def test_regression_report_and_baselines():
    targets = torch.tensor([0.1, -0.2, 0.0, 0.3])
    predictions = torch.tensor([0.2, 0.1, 0.5, 0.3])
    report = regression_report(predictions, targets)

    assert report["count"] == 4
    assert report["mae"] == pytest.approx((0.1 + 0.3 + 0.5 + 0.0) / 4)
    assert report["mse"] == pytest.approx((0.01 + 0.09 + 0.25) / 4)
    assert report["direction_accuracy"] == pytest.approx(2 / 3)

    baselines = baseline_predictions(targets, training_mean=0.05)
    torch.testing.assert_close(baselines["zero"], torch.zeros(4))
    # A zero forecast never picks a direction.
    assert math.isnan(regression_report(baselines["zero"], targets)["direction_accuracy"])
    torch.testing.assert_close(baselines["training_mean"], torch.full((4,), 0.05))
