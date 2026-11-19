"""Timestral adapter, tested with a fake forecaster: no weights are downloaded."""

from dataclasses import dataclass

import pytest
import torch

from timeseries.timestral_baseline import quantile_forecasts


LEVELS = (0.1, 0.5, 0.9)


@dataclass
class FakeForecast:
    quantiles: torch.Tensor
    quantile_levels: tuple


class FakeForecaster:
    """Returns last value + offsets, in Timestral's (B, V, H, Q) layout."""

    def __init__(self):
        self.calls = []

    def forecast(self, context, prediction_length, **options):
        self.calls.append((tuple(context.shape), prediction_length, options))
        last = context[:, -1]
        quantiles = last[:, None] + torch.tensor([-1.0, 0.0, 1.0])
        return FakeForecast(quantiles[:, None, None, :], LEVELS)


def test_quantiles_follow_windows_in_batches():
    forecaster = FakeForecaster()
    windows = torch.arange(15.0).reshape(5, 3)  # last values 2, 5, 8, 11, 14

    quantiles, levels = quantile_forecasts(forecaster, windows, batch_size=2)

    assert levels == LEVELS
    torch.testing.assert_close(quantiles[:, 1], torch.tensor([2.0, 5.0, 8.0, 11.0, 14.0]))
    assert quantiles.shape == (5, 3)
    assert [call[0] for call in forecaster.calls] == [(2, 3), (2, 3), (1, 3)]
    for _, horizon, options in forecaster.calls:
        assert horizon == 1
        assert options == {"clip_nonnegative": False}


@pytest.mark.parametrize("windows", [torch.zeros(4), torch.zeros(0, 3), torch.zeros(3, 0)])
def test_invalid_windows_are_rejected(windows):
    forecaster = FakeForecaster()
    with pytest.raises(ValueError, match="windows"):
        quantile_forecasts(forecaster, windows)
    assert forecaster.calls == []  # rejected before calling the model
