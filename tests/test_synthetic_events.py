"""Synthetic irregular events: the true drift and volatility are recoverable."""

import pytest
import torch

from market.synthetic_events import regime_events, regime_statistics


def test_each_regime_has_the_requested_drift_and_volatility():
    events = regime_events(200_000, seed=3)
    statistics = regime_statistics(events)

    assert len(statistics) == 9                         # 3 drifts × 3 volatilities
    for (drift, volatility), values in statistics.items():
        assert values["count"] > 5000
        assert values["volatility"] == pytest.approx(volatility, rel=0.05)
        # Drift error shrinks with the regime's total time: allow 4 standard errors.
        total_time = events["dt"][(events["true_drift"] == drift)
                                  & (events["true_volatility"] == volatility)].sum()
        assert abs(values["drift"] - drift) < 4 * volatility / total_time.sqrt()


def test_dt_is_irregular_and_positive_and_series_are_reproducible():
    events = regime_events(10_000, mean_dt=0.5, seed=1)
    assert (events["dt"] > 0).all()
    assert events["dt"].mean().item() == pytest.approx(0.5, rel=0.05)
    assert events["dt"].std().item() == pytest.approx(0.5, rel=0.1)   # exponential: std = mean
    again = regime_events(10_000, mean_dt=0.5, seed=1)
    for name in events:
        torch.testing.assert_close(events[name], again[name])


def test_regimes_persist_and_switch():
    events = regime_events(50_000, switch_probability=0.01, seed=2)
    changes = (events["true_volatility"][1:] != events["true_volatility"][:-1]).float().mean()
    # A switch redraws among 3 values, so about 2/3 of switches change the volatility.
    assert 0.004 < changes.item() < 0.01


@pytest.mark.parametrize("options", [{"count": 0}, {"count": 10, "mean_dt": 0.0},
                                     {"count": 10, "volatilities": (1.0, 0.0)},
                                     {"count": 10, "switch_probability": 1.5}])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        regime_events(**options)
