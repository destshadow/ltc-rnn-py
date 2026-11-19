"""Synthetic AR(1) series: the signal really has the requested strength."""

import pytest
import torch

from timeseries.synthetic import ar1_prices
from timeseries.windows import log_returns


def lag1_correlation(values):
    return torch.corrcoef(torch.stack([values[:-1], values[1:]]))[0, 1].item()


@pytest.mark.parametrize("phi", [0.0, 0.3, -0.5])
def test_returns_have_the_requested_autocorrelation(phi):
    starts, prices = ar1_prices(20000, phi=phi, seed=1)
    returns = log_returns(prices)

    assert len(starts) == len(prices) == 20000
    assert all(later - earlier == starts[1] - starts[0] for earlier, later in zip(starts, starts[1:]))
    assert lag1_correlation(returns) == pytest.approx(phi, abs=0.03)


def test_same_seed_same_series_and_invalid_phi():
    torch.testing.assert_close(ar1_prices(50, phi=0.2, seed=3)[1], ar1_prices(50, phi=0.2, seed=3)[1])
    with pytest.raises(ValueError, match="phi"):
        ar1_prices(50, phi=1.0)
