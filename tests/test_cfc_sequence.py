"""CfC sequences: equivalence with step-by-step updates and per-step Δt."""

import pytest
import torch

from cfc.cell import CfcCell
from cfc.config import CfcConfig
from cfc.sequence import run_cfc_sequence


@pytest.fixture
def cell():
    torch.manual_seed(1)
    return CfcCell(CfcConfig(input_size=2, hidden_size=4))


def test_sequence_equals_manual_steps_with_irregular_dt(cell):
    inputs = torch.randn(3, 6, 2)
    dt = torch.rand(3, 6) * 2                      # a different interval at every step
    history, final = run_cfc_sequence(cell, inputs, dt=dt)

    state = cell.initial_state(3)
    for index in range(6):
        state = cell(inputs[:, index], state, dt=dt[:, index])
        torch.testing.assert_close(history[:, index], state)
    torch.testing.assert_close(final, state)


def test_scalar_dt_and_history_options(cell):
    inputs = torch.randn(2, 5, 2)
    history, final = run_cfc_sequence(cell, inputs, dt=0.5)
    none, same_final = run_cfc_sequence(cell, inputs, dt=torch.full((2, 5), 0.5),
                                        collect_history=False)
    assert history.shape == (2, 5, 4) and none is None
    torch.testing.assert_close(final, same_final)


def test_continuing_from_a_state_equals_one_long_run(cell):
    inputs, dt = torch.randn(2, 8, 2), torch.rand(2, 8)
    _, whole = run_cfc_sequence(cell, inputs, dt=dt, collect_history=False)
    _, half = run_cfc_sequence(cell, inputs[:, :4], dt=dt[:, :4], collect_history=False)
    _, rest = run_cfc_sequence(cell, inputs[:, 4:], dt=dt[:, 4:], initial_state=half,
                               collect_history=False)
    torch.testing.assert_close(rest, whole)


@pytest.mark.parametrize("inputs,dt", [
    (torch.randn(2, 2), 0.1),                      # not [batch, time, inputs]
    (torch.randn(0, 3, 2), 0.1),                   # empty batch
    (torch.randn(2, 3, 2), torch.ones(2, 4)),      # dt with wrong length
])
def test_invalid_shapes_are_rejected(cell, inputs, dt):
    with pytest.raises(ValueError):
        run_cfc_sequence(cell, inputs, dt=dt)
