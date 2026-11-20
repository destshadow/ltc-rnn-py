"""Linear AR(1) baseline: coefficients, alignment with windows, no pairs across gaps."""

from datetime import datetime, timedelta

import pytest
import torch

from timeseries.calendar import split_segments
from timeseries.dataset import Piece, make_dataset, returns_by_segment, window_targets
from timeseries.linear_baseline import fit_ar1, window_last_returns
from timeseries.synthetic import ar1_prices
from timeseries.windows import Standardizer

DAY = timedelta(days=1)


def test_recovers_the_ar1_coefficient_of_a_synthetic_series():
    starts, prices = ar1_prices(20000, phi=0.3, seed=4)
    pieces = returns_by_segment([(starts, prices)], period=DAY)
    slope, intercept = fit_ar1(pieces)
    assert slope == pytest.approx(0.3, abs=0.03)
    assert intercept == pytest.approx(0.0, abs=1e-3)


def test_pairs_never_cross_pieces():
    # Inside each piece tomorrow = 2·today exactly; across the gap it would not hold.
    first = Piece(torch.tensor([1.0, 2.0, 4.0]), torch.ones(3), [datetime(2024, 1, k) for k in (2, 3, 4)])
    second = Piece(torch.tensor([-1.0, -2.0]), torch.ones(2), [datetime(2024, 1, k) for k in (8, 9)])
    assert fit_ar1([first, second]) == pytest.approx((2.0, 0.0))


def test_last_returns_are_aligned_with_targets():
    starts = [datetime(2024, 1, 1) + k * DAY for k in range(15) if k != 7]
    prices = torch.linspace(100, 130, len(starts), dtype=torch.float64) + torch.rand(len(starts))
    pieces = returns_by_segment(split_segments(starts, prices, period=DAY, calendar="continuous"),
                                period=DAY)
    scaler = Standardizer.fit(torch.cat([piece.returns for piece in pieces]))
    inputs, _ = make_dataset(pieces, scaler=scaler, lookback=3)

    last = window_last_returns(pieces, lookback=3)
    assert len(last) == len(window_targets(pieces, lookback=3)) == len(inputs)
    torch.testing.assert_close(last, scaler.inverse(inputs[:, -1, 0].double()), rtol=0, atol=1e-6)


def test_degenerate_inputs_are_rejected():
    single = Piece(torch.tensor([1.0]), torch.ones(1), [datetime(2024, 1, 2)])
    with pytest.raises(ValueError):
        fit_ar1([single])
    constant = Piece(torch.ones(5), torch.ones(5), [datetime(2024, 1, k) for k in range(2, 7)])
    with pytest.raises(ValueError, match="costanti"):
        fit_ar1([constant])
