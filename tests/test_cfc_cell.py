"""CfC cell: shapes, per-sample Δt, bounded state, gradients and input checks."""

import pytest
import torch

from cfc.cell import CfcCell
from cfc.config import CfcConfig


def make_cell(**options):
    torch.manual_seed(0)
    return CfcCell(CfcConfig(input_size=3, hidden_size=5, **options))


def test_shapes_and_bounded_state():
    cell = make_cell(backbone_layers=2)
    state = cell(torch.randn(4, 3) * 50, cell.initial_state(4), dt=0.7)
    assert state.shape == (4, 5)
    assert state.abs().max() <= 1.0                     # convex mix of tanh outputs


def test_scalar_dt_equals_constant_dt_vector():
    cell = make_cell()
    inputs, state = torch.randn(4, 3), torch.randn(4, 5).tanh()
    torch.testing.assert_close(cell(inputs, state, dt=0.3),
                               cell(inputs, state, dt=torch.full((4,), 0.3)))


def test_each_sample_uses_its_own_dt():
    cell = make_cell()
    inputs, state = torch.randn(1, 3).repeat(2, 1), torch.zeros(2, 5)
    out = cell(inputs, state, dt=torch.tensor([0.1, 5.0]))
    single = cell(inputs[:1], state[:1], dt=5.0)

    assert not torch.allclose(out[0], out[1])           # same input, different Δt
    torch.testing.assert_close(out[1:], single)          # sample 1 behaves as if alone


def test_gradients_reach_every_parameter():
    cell = make_cell()
    cell(torch.randn(4, 3), cell.initial_state(4), dt=torch.rand(4)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())


def test_weight_names_match_the_authors_cell():
    names = {name.split(".")[0] for name, _ in make_cell().named_parameters()}
    assert names == {"backbone", "ff1", "ff2", "time_a", "time_b"}


@pytest.mark.parametrize("inputs,state,dt", [
    (torch.randn(4, 2), torch.zeros(4, 5), 0.1),         # wrong input size
    (torch.tensor(1.0), torch.zeros(1, 5), 0.1),         # scalar input
    (torch.randn(4, 3), torch.zeros(3, 5), 0.1),         # wrong batch in state
    (torch.randn(4, 3), torch.zeros(4, 5), -0.1),        # negative Δt
    (torch.randn(4, 3), torch.zeros(4, 5), torch.ones(3)),
    (torch.randn(4, 3), torch.zeros(4, 5), float("nan")),
])
def test_invalid_inputs_are_rejected(inputs, state, dt):
    with pytest.raises(ValueError):
        make_cell()(inputs, state, dt=dt)


@pytest.mark.parametrize("field", ["input_size", "hidden_size", "backbone_units", "backbone_layers"])
def test_config_rejects_non_positive_sizes(field):
    values = {"input_size": 3, "hidden_size": 5, field: 0}
    with pytest.raises(ValueError):
        CfcConfig(**values)
