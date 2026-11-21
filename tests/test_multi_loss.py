"""Weighted multi-output loss and training on [n, outputs] targets."""

import pytest
import torch

from cfc.config import CfcConfig
from timeseries.model import MultiOutputForecaster
from training.epoch import train_epoch
from training.multi_loss import MultiOutputLoss

OUTPUTS = ("drift", "volatility")


def test_weighted_mean_and_per_output_values():
    predictions = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    targets = torch.zeros(2, 2)
    loss = MultiOutputLoss(OUTPUTS, {"drift": 1.0, "volatility": 3.0})

    assert loss.per_output(predictions, targets) == pytest.approx({"drift": 5.0, "volatility": 0.0})
    assert loss(predictions, targets).item() == pytest.approx(0.25 * 5.0)   # weights 1/4, 3/4
    assert MultiOutputLoss(OUTPUTS)(predictions, targets).item() == pytest.approx(2.5)


@pytest.mark.parametrize("weights", [{"drift": 1.0}, {"drift": -1.0, "volatility": 1.0},
                                     {"drift": 0.0, "volatility": 0.0}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        MultiOutputLoss(OUTPUTS, weights)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        MultiOutputLoss(OUTPUTS)(torch.zeros(3, 2), torch.zeros(3, 3))


def test_training_reduces_the_loss_on_two_learnable_targets():
    torch.manual_seed(0)
    inputs = torch.stack([torch.randn(256, 8), torch.ones(256, 8)], dim=2)
    # drift: last value; volatility: mean |value| of the window — both computable from inputs.
    targets = torch.stack([inputs[:, -1, 0], inputs[:, :, 0].abs().mean(dim=1)], dim=1)
    targets = (targets - targets.mean(0)) / targets.std(0)

    model = MultiOutputForecaster(CfcConfig(2, 16, backbone_units=32), OUTPUTS, elapsed_channel=1)
    loss = MultiOutputLoss(OUTPUTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = loss.per_output(predict_all(model, inputs), targets)
    for _ in range(40):
        train_epoch(model, optimizer, loss, inputs, targets, dt=0.1, batch_size=64)
    after = loss.per_output(predict_all(model, inputs), targets)

    assert all(after[name] < 0.5 * before[name] for name in OUTPUTS), (before, after)


def predict_all(model, inputs):
    with torch.no_grad():
        return model(inputs, dt=0.1)
