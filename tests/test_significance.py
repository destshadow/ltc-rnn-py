"""Diebold–Mariano test: sign, size and calibration on known cases."""

import math

import pytest
import torch

from timeseries.significance import diebold_mariano


def test_clearly_better_forecast_is_significant():
    generator = torch.Generator().manual_seed(0)
    targets = torch.randn(2000, generator=generator)
    good = targets + 0.5 * torch.randn(2000, generator=generator)
    bad = torch.zeros(2000)

    result = diebold_mariano((good - targets) ** 2, (bad - targets) ** 2)
    assert result["mean_difference"] < 0 and result["statistic"] < -5
    assert result["p_value"] < 1e-6


def test_equivalent_forecasts_are_rejected_at_about_the_nominal_rate():
    # Two forecasts with the same error distribution: p < 0.05 about 5 % of the time.
    rejections = 0
    for seed in range(200):
        generator = torch.Generator().manual_seed(seed)
        loss_a = torch.randn(500, generator=generator) ** 2
        loss_b = torch.randn(500, generator=generator) ** 2
        rejections += diebold_mariano(loss_a, loss_b)["p_value"] < 0.05
    assert 2 <= rejections <= 22   # wide band around 10 expected


def test_without_lags_matches_the_plain_t_statistic():
    loss_a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss_b = torch.tensor([1.5, 1.5, 3.5, 3.5])
    d = loss_a - loss_b                                       # -0.5, 0.5, -0.5, 0.5
    expected = d.mean() / math.sqrt(d.var(unbiased=False) / 4)
    assert diebold_mariano(loss_a, loss_b, lags=0)["statistic"] == pytest.approx(expected.item())


@pytest.mark.parametrize("loss_a,loss_b,lags", [
    (torch.ones(10), torch.ones(9), 1),          # shape mismatch
    (torch.ones(10), torch.ones(10), -1),        # invalid lags
    (torch.ones(3), torch.ones(3), 5),           # too few examples
    (torch.ones(10), torch.zeros(10), 1),        # constant difference
])
def test_invalid_inputs_are_rejected(loss_a, loss_b, lags):
    with pytest.raises(ValueError):
        diebold_mariano(loss_a, loss_b, lags=lags)
