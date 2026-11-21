"""Multi-output forecaster: one shared cell, several named estimates."""

import pytest
import torch

from cfc.config import CfcConfig
from ltc.config import LTCConfig
from timeseries.model import MultiOutputForecaster, ReturnForecaster

OUTPUTS = ("drift", "volatility")


def inputs(batch=4, steps=6):
    return torch.stack([torch.randn(batch, steps), torch.rand(batch, steps) + 0.5], dim=2)


@pytest.mark.parametrize("config,channel", [(CfcConfig(2, 5, backbone_units=8), 1), (LTCConfig(2, 5), None)])
def test_shapes_names_and_shared_gradients(config, channel):
    torch.manual_seed(0)
    model = MultiOutputForecaster(config, OUTPUTS, elapsed_channel=channel)
    predictions = model(inputs(), dt=0.1)

    assert predictions.shape == (4, 2)
    named = model.named(predictions)
    assert list(named) == list(OUTPUTS)
    torch.testing.assert_close(named["volatility"], predictions[:, 1])

    # Each output alone reaches the shared cell.
    for name in OUTPUTS:
        model.zero_grad()
        model.named(model(inputs(), dt=0.1))[name].sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cell.parameters())


def test_single_output_matches_the_scalar_forecaster():
    torch.manual_seed(0)
    config = CfcConfig(2, 5, backbone_units=8)
    scalar = ReturnForecaster(config, elapsed_channel=1)
    multi = MultiOutputForecaster(config, ("drift",), elapsed_channel=1)
    multi.load_state_dict(scalar.state_dict())

    batch = inputs()
    torch.testing.assert_close(multi(batch, dt=0.1)[:, 0], scalar(batch, dt=0.1))


@pytest.mark.parametrize("outputs", [(), ("drift", "drift"), ("",), (3,)])
def test_invalid_output_names_are_rejected(outputs):
    with pytest.raises(ValueError):
        MultiOutputForecaster(CfcConfig(2, 3), outputs, elapsed_channel=1)


def test_named_rejects_wrong_shapes():
    model = MultiOutputForecaster(CfcConfig(2, 3), OUTPUTS, elapsed_channel=1)
    with pytest.raises(ValueError):
        model.named(torch.zeros(4, 3))
