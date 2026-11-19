"""Quantile metrics: pinball loss and interval coverage on hand-checked values."""

import pytest
import torch

from timeseries.quantile_metrics import interval_coverage, pinball_loss


LEVELS = (0.1, 0.5, 0.9)


def test_pinball_loss_matches_hand_computation():
    quantiles = torch.tensor([[-1.0, 0.0, 1.0]])
    # y = 2: errors 3, 2, 1 → 0.1·3, 0.5·2, 0.9·1 → mean (0.3 + 1.0 + 0.9) / 3
    assert pinball_loss(quantiles, LEVELS, torch.tensor([2.0])) == pytest.approx(2.2 / 3)
    # y = -2: errors -1, -2, -3 → 0.9·1, 0.5·2, 0.1·3
    assert pinball_loss(quantiles, LEVELS, torch.tensor([-2.0])) == pytest.approx(2.2 / 3)


def test_median_term_is_half_the_absolute_error():
    targets = torch.tensor([0.5, -1.5, 2.0])
    quantiles = torch.zeros(3, 1)
    assert pinball_loss(quantiles, (0.5,), targets) == pytest.approx(0.5 * targets.abs().mean().item())


def test_interval_coverage_counts_targets_inside_the_band():
    quantiles = torch.tensor([[-1.0, 0.0, 1.0]] * 4)
    targets = torch.tensor([0.0, 1.0, -1.5, 3.0])  # inside, on the edge, below, above

    report = interval_coverage(quantiles, LEVELS, targets, lower=0.1, upper=0.9)
    assert report == pytest.approx({"nominal": 0.8, "coverage": 0.5, "mean_width": 2.0})


@pytest.mark.parametrize("quantiles,levels,targets,options", [
    (torch.zeros(2, 3), LEVELS, torch.zeros(3), {}),            # shape mismatch
    (torch.zeros(1, 2), (0.0, 0.5), torch.zeros(1), {}),        # invalid level
    (torch.zeros(1, 3), LEVELS, torch.zeros(1), {"lower": 0.2, "upper": 0.9}),
    (torch.zeros(1, 3), LEVELS, torch.zeros(1), {"lower": 0.9, "upper": 0.1}),
])
def test_invalid_inputs_are_rejected(quantiles, levels, targets, options):
    with pytest.raises(ValueError):
        if options:
            interval_coverage(quantiles, levels, targets, **options)
        else:
            pinball_loss(quantiles, levels, targets)
