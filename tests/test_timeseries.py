"""Time-series data handling: alignment, chronological order and no leakage."""

import json
import math
from datetime import datetime, timedelta

import pytest
import torch

from ltc.config import LTCConfig
from timeseries.calendar import elapsed_periods, expected_periods, split_segments
from timeseries.dataset import make_dataset, returns_by_segment, split_pieces
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
    day = timedelta(days=1)
    starts, prices = resample_last(iter_csv_prices(path), period=day)
    torch.testing.assert_close(elapsed_periods(starts, period=day),
                               torch.tensor([2.0], dtype=torch.float64))
    assert len(split_segments(starts, prices, period=day, calendar="continuous")) == 2


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
    # A zero forecast never picks a direction: the metric is absent, not NaN,
    # so the report stays valid strict JSON.
    zero_report = regression_report(baselines["zero"], targets)
    assert zero_report["direction_accuracy"] is None
    assert json.loads(json.dumps(zero_report, allow_nan=False)) == zero_report
    torch.testing.assert_close(baselines["training_mean"], torch.full((4,), 0.05))


def test_windows_with_several_inputs_keep_time_alignment():
    series = torch.arange(10.0).reshape(5, 2)
    targets = torch.arange(100.0, 105.0)
    inputs, window_targets = make_windows(series, lookback=2, targets=targets)

    assert inputs.shape == (3, 2, 2)
    for index in range(3):
        torch.testing.assert_close(inputs[index], series[index:index + 2])
    torch.testing.assert_close(window_targets, targets[2:])

    with pytest.raises(ValueError, match="targets"):
        make_windows(series, lookback=2)


# --- Step 1 acceptance criteria: calendar segments in the training data ---

DAY = timedelta(days=1)
MISSING_TUESDAY = datetime(2024, 1, 9)


def weekday_series():
    """Weekday closes from Mon 2024-01-01 to Fri 01-19, Tue 01-09 missing.

    Returns differ day by day; across the missing day the price jumps x100,
    so any return computed across the gap would be log(100).
    """
    starts, prices = [], []
    day, price = datetime(2024, 1, 1), 100.0
    while day <= datetime(2024, 1, 19):
        if day.weekday() < 5 and day != MISSING_TUESDAY:
            price *= 100.0 if day == MISSING_TUESDAY + DAY else 1 + day.day / 1000
            starts.append(day)
            prices.append(price)
        day += DAY
    return starts, torch.tensor(prices, dtype=torch.float64)


def pieces_for(calendar):
    starts, prices = weekday_series()
    segments = split_segments(starts, prices, period=DAY, calendar=calendar)
    return returns_by_segment(segments, period=DAY)


@pytest.mark.parametrize("calendar,expected_elapsed", [
    # Weekend expected: Fri -> Mon stays in the piece, lasting 3 days.
    ("weekdays", [[1, 1, 1, 1, 3], [1, 1, 3, 1, 1, 1, 1]]),
    # Every day expected: weekends and the Tuesday all cut the series.
    ("continuous", [[1, 1, 1, 1], [1, 1], [1, 1, 1, 1]]),
])
def test_criterion1_no_return_or_window_crosses_missing_data(calendar, expected_elapsed):
    pieces = pieces_for(calendar)

    assert [piece.elapsed.tolist() for piece in pieces] == expected_elapsed
    for piece in pieces:
        assert MISSING_TUESDAY not in piece.end_times
        assert piece.returns.abs().max() < math.log(2)

    scaler = Standardizer.fit(torch.cat([piece.returns for piece in pieces]))
    inputs, targets = make_dataset(pieces, scaler=scaler, lookback=2)
    restored = scaler.inverse(torch.cat([inputs[:, :, 0].flatten(), targets]).double())
    assert restored.abs().max() < math.log(2)


@pytest.mark.parametrize("calendar", ["weekdays", "continuous"])
def test_criterion1_windows_never_join_two_pieces(calendar):
    # Returns are all small, so only an explicit reference catches windows
    # built after concatenating the pieces.
    pieces = pieces_for(calendar)
    lookback = 2
    scaler = Standardizer.fit(torch.cat([piece.returns for piece in pieces]))
    inputs, targets = make_dataset(pieces, scaler=scaler, lookback=lookback)

    expected_inputs, expected_targets = [], []
    for piece in pieces:
        standardized = scaler.transform(piece.returns)
        features = torch.stack([standardized, piece.elapsed], dim=1)
        for k in range(len(piece.returns) - lookback):
            expected_inputs.append(features[k:k + lookback])
            expected_targets.append(standardized[k + lookback])

    assert len(targets) == sum(max(0, len(p.returns) - lookback) for p in pieces)
    torch.testing.assert_close(inputs, torch.stack(expected_inputs).float())
    torch.testing.assert_close(targets, torch.stack(expected_targets).float())


def test_criterion2_elapsed_is_aligned_with_its_return():
    starts, prices = weekday_series()
    pieces = pieces_for("weekdays")

    for piece in pieces:
        for return_value, elapsed, end in zip(piece.returns, piece.elapsed, piece.end_times):
            index = starts.index(end)
            assert return_value == torch.log(prices[index] / prices[index - 1])
            assert elapsed == (end - starts[index - 1]) / DAY

    piece = pieces[1]
    scaler = Standardizer.fit(piece.returns)
    inputs, targets = make_dataset([piece], scaler=scaler, lookback=2)
    for k in range(len(targets)):
        torch.testing.assert_close(
            scaler.inverse(inputs[k, :, 0].double()), piece.returns[k:k + 2],
        )
        torch.testing.assert_close(inputs[k, :, 1].double(), piece.elapsed[k:k + 2])
        torch.testing.assert_close(
            scaler.inverse(targets[k].double()), piece.returns[k + 2],
        )


def daily_series(missing_day=50, days=100, shock=1.0):
    """Continuous daily closes with one missing day; shock scales late returns."""
    start = datetime(2024, 1, 1)
    starts, prices, price = [], [], 100.0
    for index in range(days):
        if index == missing_day:
            continue
        step = 1 + (index % 7 - 3) / 100
        price *= step ** (shock if index >= 90 else 1.0)
        starts.append(start + index * DAY)
        prices.append(price)
    return starts, torch.tensor(prices, dtype=torch.float64)


def split_for(**series_options):
    starts, prices = daily_series(**series_options)
    segments = split_segments(starts, prices, period=DAY, calendar="continuous")
    return split_pieces(
        returns_by_segment(segments, period=DAY),
        validation_fraction=0.15, test_fraction=0.15,
    )


def test_criterion3_normalization_uses_only_training():
    training, validation, _ = split_for()
    scaler = Standardizer.fit(torch.cat([piece.returns for piece in training]))

    # Changing only the last (test) returns leaves the training statistics unchanged.
    shocked_training, _, shocked_test = split_for(shock=5.0)
    assert Standardizer.fit(torch.cat([p.returns for p in shocked_training])) == scaler
    assert not torch.equal(shocked_test[-1].returns, split_for()[2][-1].returns)

    # Validation features are scaled with the training statistics, not refit.
    inputs, _ = make_dataset(validation, scaler=scaler, lookback=3)
    expected = scaler.transform(validation[0].returns[:3]).float()
    torch.testing.assert_close(inputs[0, :, 0], expected)


def test_criterion4_split_is_chronological_and_complete():
    parts = split_for()
    counts = [sum(len(piece.returns) for piece in part) for part in parts]

    # 49 + 48 returns from the two segments around the missing day.
    assert counts == [69, 14, 14]

    times = [[time for piece in part for time in piece.end_times] for part in parts]
    assert max(times[0]) < min(times[1]) and max(times[1]) < min(times[2])
    for part in parts:
        for piece in part:
            ends = piece.end_times
            assert all(later - earlier == DAY for earlier, later in zip(ends, ends[1:]))
