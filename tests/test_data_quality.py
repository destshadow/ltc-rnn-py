"""Data-quality statistics on series with known properties."""

import pytest
import torch

from timeseries.data_quality import return_stats
from timeseries.synthetic import ar1_prices
from timeseries.windows import log_returns


@pytest.mark.parametrize("phi", [0.3, -0.2])
def test_autocorrelation_sign_and_size_are_recovered(phi):
    returns = log_returns(ar1_prices(20000, phi=phi, seed=2)[1])
    assert return_stats([returns])["lag1_autocorrelation"] == pytest.approx(phi, abs=0.03)


def test_basic_statistics_and_extremes():
    values = torch.cat([torch.tensor([1.0, -1.0] * 500), torch.tensor([50.0])])
    stats = return_stats([values])

    assert stats["count"] == 1001
    assert stats["max_abs"] == 50.0
    assert stats["extreme_share"] == pytest.approx(1 / 1001)   # only the spike
    assert stats["lag1_autocorrelation"] < 0                     # alternating signs


@pytest.mark.parametrize("runs", [[torch.zeros(5)], [torch.ones(2)], [torch.ones(3, 2)], []])
def test_invalid_inputs_are_rejected(runs):
    with pytest.raises(ValueError):
        return_stats(runs)


def test_no_lag1_pair_across_a_gap():
    # Within each run the signs alternate (negative pairs); the only positive
    # products would be across the gap, joining 1.0 with 1.0.
    first, second = torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0])
    joined = return_stats([torch.cat([first, second])])["lag1_autocorrelation"]
    separated = return_stats([first, second])["lag1_autocorrelation"]

    assert separated == pytest.approx(-2 / 4)       # two inner pairs of −1, Σ squares = 4
    assert joined == pytest.approx((-1 + 1 - 1) / 4)  # includes the pair across the gap


def test_singleton_runs_are_not_reported_as_zero_autocorrelation():
    from experiments.data_quality_report import safe_stats

    runs = [torch.tensor([0.01]), torch.tensor([-0.02]), torch.tensor([0.03])]
    with pytest.raises(ValueError, match="coppia"):
        return_stats(runs)

    stats = safe_stats(runs)
    assert stats["count"] == 3
    assert "lag1_autocorrelation" not in stats and "coppia" in stats["unavailable"]
