"""Event windows: alignment with the next event and naive window estimators."""

import pytest
import torch

from market.event_windows import event_windows
from market.synthetic_events import regime_events


def test_windows_targets_and_naive_estimates_are_aligned():
    events = regime_events(50, seed=0)
    windows = event_windows(events, lookback=5)

    assert windows["inputs"].shape == (45, 5, 2)
    for k in (0, 17, 44):
        torch.testing.assert_close(windows["inputs"][k, :, 0], events["returns"][k:k + 5])
        torch.testing.assert_close(windows["inputs"][k, :, 1], events["dt"][k:k + 5])
        # Targets are the true values of the event right AFTER the window.
        assert windows["targets"][k, 0] == events["true_drift"][k + 5]
        assert windows["targets"][k, 1] == events["true_volatility"][k + 5]
        r, dt = events["returns"][k:k + 5], events["dt"][k:k + 5]
        drift = r.sum() / dt.sum()
        assert windows["naive"][k, 0].item() == pytest.approx(drift.item())
        assert windows["naive"][k, 1].item() == pytest.approx(((r - drift * dt) / dt.sqrt()).std().item())


def test_naive_volatility_is_close_to_the_truth_in_a_single_regime():
    events = regime_events(20_000, drifts=(0.0,), volatilities=(1.5,), seed=4)
    naive = event_windows(events, lookback=200)["naive"]
    assert naive[:, 1].mean().item() == pytest.approx(1.5, rel=0.03)


@pytest.mark.parametrize("lookback", [1, 60])
def test_invalid_lookback_is_rejected(lookback):
    with pytest.raises(ValueError):
        event_windows(regime_events(50, seed=0), lookback=lookback)
