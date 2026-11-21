"""Book windows: four aligned targets, no informed flag in the inputs, naive estimators."""

import pytest
import torch

from market.book_windows import OUTPUTS, book_windows
from market.synthetic_book import book_events


def test_inputs_targets_and_naive_estimates_are_aligned():
    events = book_events(60, seed=0)
    windows = book_windows(events, lookback=6)
    assert windows["inputs"].shape == (54, 6, 4) and windows["targets"].shape == (54, 4)
    assert OUTPUTS == ("drift", "volatility", "toxicity", "k")

    k = 20
    for channel, name in enumerate(("returns", "dt", "side", "reach")):
        torch.testing.assert_close(windows["inputs"][k, :, channel], events[name][k:k + 6].double())
    expected = [events[name][k + 6] for name in ("true_drift", "true_volatility", "true_toxicity", "true_k")]
    torch.testing.assert_close(windows["targets"][k], torch.stack(expected).double())

    side, returns = events["side"][k:k + 6], events["returns"][k:k + 6]
    agree = (side[:-1] == torch.where(returns[1:] >= 0, 1.0, -1.0)).double().mean()
    assert windows["naive"][k, 2].item() == pytest.approx(min(max(2 * agree.item() - 1, 0.0), 1.0))
    assert windows["naive"][k, 3].item() == pytest.approx(1 / events["reach"][k:k + 6].mean().item())


def test_informed_flag_never_enters_the_inputs():
    events = book_events(80, seed=1)
    flipped = dict(events, informed=~events["informed"])
    torch.testing.assert_close(book_windows(events, lookback=5)["inputs"],
                               book_windows(flipped, lookback=5)["inputs"])


def test_naive_estimators_recover_toxicity_and_k_in_a_single_regime():
    events = book_events(40_000, toxicities=(0.4,), ks=(1.5,), seed=3)
    naive = book_windows(events, lookback=400)["naive"]
    assert naive[:, 2].mean().item() == pytest.approx(0.4, abs=0.03)
    assert naive[:, 3].mean().item() == pytest.approx(1.5, rel=0.03)
