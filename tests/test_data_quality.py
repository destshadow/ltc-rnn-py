"""Data-quality statistics on series with known properties."""

import pytest
import torch

from timeseries.data_quality import return_stats
from timeseries.synthetic import ar1_prices
from timeseries.windows import log_returns


@pytest.mark.parametrize("phi", [0.3, -0.2])
def test_autocorrelation_sign_and_size_are_recovered(phi):
    returns = log_returns(ar1_prices(20000, phi=phi, seed=2)[1])
    assert return_stats(returns)["lag1_autocorrelation"] == pytest.approx(phi, abs=0.03)


def test_basic_statistics_and_extremes():
    values = torch.cat([torch.tensor([1.0, -1.0] * 500), torch.tensor([50.0])])
    stats = return_stats(values)

    assert stats["count"] == 1001
    assert stats["max_abs"] == 50.0
    assert stats["extreme_share"] == pytest.approx(1 / 1001)   # only the spike
    assert stats["lag1_autocorrelation"] < 0                     # alternating signs


@pytest.mark.parametrize("values", [torch.zeros(5), torch.ones(2), torch.ones(3, 2)])
def test_invalid_inputs_are_rejected(values):
    with pytest.raises(ValueError):
        return_stats(values)
