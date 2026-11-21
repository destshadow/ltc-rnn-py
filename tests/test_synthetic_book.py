"""Synthetic book events: informed share equals toxicity, reach has rate k."""

import math

import pytest
import torch

from market.synthetic_book import book_events


@pytest.fixture(scope="module")
def events():
    return book_events(200_000, seed=5)


def test_informed_share_and_side_alignment_follow_the_toxicity(events):
    next_sign = torch.where(torch.cat([events["returns"][1:], events["returns"][-1:]]) >= 0, 1.0, -1.0)
    for toxicity in (0.1, 0.3, 0.6):
        mask = events["true_toxicity"] == toxicity
        assert mask.sum() > 20_000
        assert events["informed"][mask].double().mean().item() == pytest.approx(toxicity, abs=0.01)
        # Informed always agree with the next move, uninformed half of the time.
        agreement = (events["side"][mask] == next_sign[mask]).double().mean().item()
        assert agreement == pytest.approx(toxicity + (1 - toxicity) / 2, abs=0.01)


def test_reach_is_exponential_with_rate_k(events):
    for k in (0.5, 1.0, 2.0):
        reach = events["reach"][events["true_k"] == k]
        assert reach.mean().item() == pytest.approx(1 / k, rel=0.02)
        # Probability that a limit order at distance δ is filled: exp(−k·δ).
        delta = 1.0
        assert (reach >= delta).double().mean().item() == pytest.approx(math.exp(-k * delta), abs=0.01)


def test_sides_are_plus_minus_one_and_series_are_reproducible():
    first, second = book_events(5_000, seed=2), book_events(5_000, seed=2)
    assert set(first["side"].unique().tolist()) <= {-1.0, 1.0}
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    # The price path is the same as regime_events with the same seed (independent streams).
    from market.synthetic_events import regime_events
    torch.testing.assert_close(first["returns"], regime_events(5_000, seed=2)["returns"])


@pytest.mark.parametrize("options", [{"toxicities": (0.2, 1.5)}, {"toxicities": ()},
                                     {"ks": (1.0, 0.0)}, {"ks": ()}])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        book_events(100, **options)
