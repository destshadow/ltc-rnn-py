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


def test_newey_west_scale_on_a_hand_computed_case():
    # d alternates 0, 2 → centred ±1. With every autocovariance divided by n = 8 and
    # Bartlett weights for 5 lags the long-run variance is 0.125; mean(d) = 1, so
    # statistic = 1 / sqrt(0.125 / 8) = 8.
    loss_a = torch.tensor([0.0, 2.0] * 4)
    result = diebold_mariano(loss_a, torch.zeros(8), lags=5)
    assert result["statistic"] == pytest.approx(8.0)
    assert result["p_value"] == pytest.approx(math.erfc(8 / math.sqrt(2)))


def test_unavailable_test_is_reported_instead_of_raising():
    from timeseries.significance import diebold_mariano_or_unavailable

    few = diebold_mariano_or_unavailable(torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3))
    identical = diebold_mariano_or_unavailable(torch.ones(20), torch.ones(20))
    for result in (few, identical):
        assert result["p_value"] is None and result["unavailable"]
