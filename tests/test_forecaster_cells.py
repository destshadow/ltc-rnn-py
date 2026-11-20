"""Forecaster with LTC or CfC: the CfC takes its Δt from the elapsed-time input."""

import pytest
import torch

from cfc.config import CfcConfig
from timeseries.model import ReturnForecaster, make_forecaster


def build(cell):
    torch.manual_seed(0)
    return make_forecaster(cell, input_size=2, hidden_size=4, substeps=2, backbone_units=8,
                           elapsed_channel=1)


@pytest.mark.parametrize("cell", ["ltc", "cfc", "lstm"])
def test_shapes_and_gradients(cell):
    config, model = build(cell)
    inputs = torch.stack([torch.randn(5, 7), torch.ones(5, 7)], dim=2)
    output = model(inputs, dt=0.1)
    assert output.shape == (5,)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert config.hidden_size == 4


def test_cfc_reads_the_elapsed_time_as_dt():
    _, model = build("cfc")
    returns = torch.randn(1, 6)
    regular = torch.stack([returns, torch.ones(1, 6)], dim=2)
    weekend = regular.clone()
    weekend[0, 3, 1] = 3.0                    # Fri -> Mon: three periods instead of one

    # Same returns, different elapsed time: the CfC output changes.
    assert not torch.allclose(model(regular, dt=0.1), model(weekend, dt=0.1))
    # dt scales the elapsed time: a different dt on the same inputs changes the output.
    assert not torch.allclose(model(regular, dt=0.1), model(regular, dt=1.0))


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError, match="elapsed_channel"):
        ReturnForecaster(CfcConfig(input_size=2, hidden_size=3))
    with pytest.raises(ValueError, match="cell"):
        make_forecaster("gru", input_size=2, hidden_size=3, substeps=1, backbone_units=4,
                        elapsed_channel=1)
    with pytest.raises(TypeError):
        ReturnForecaster({"hidden_size": 3})


def test_lstm_ignores_dt_and_matches_a_plain_lstm():
    config, model = build("lstm")
    inputs = torch.stack([torch.randn(3, 5), torch.ones(3, 5)], dim=2)
    torch.testing.assert_close(model(inputs, dt=0.1), model(inputs, dt=7.0))   # no continuous time

    _, (hidden, _) = model.lstm(inputs)
    torch.testing.assert_close(model(inputs, dt=0.1), model.readout(hidden[-1]).squeeze(-1))
    assert (config.input_size, config.hidden_size) == (2, 4)
