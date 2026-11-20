"""Our CfC cell against the authors' implementation (ncps), with identical weights.

Skipped when ncps is not installed: python -m pip install -e '.[authors]'
"""

import pytest
import torch

ncps_torch = pytest.importorskip("ncps.torch")

from cfc.cell import CfcCell  # noqa: E402
from cfc.config import CfcConfig  # noqa: E402


@pytest.mark.parametrize("layers", [1, 2])
def test_same_outputs_and_gradients_as_the_authors(layers):
    torch.manual_seed(0)
    theirs = ncps_torch.CfCCell(3, 5, mode="default", backbone_activation="lecun_tanh",
                                backbone_units=16, backbone_layers=layers)
    ours = CfcCell(CfcConfig(input_size=3, hidden_size=5, backbone_units=16, backbone_layers=layers))
    ours.load_state_dict(theirs.state_dict())       # same parameter names and shapes

    inputs, state = torch.randn(4, 3), torch.randn(4, 5).tanh()
    dt = torch.tensor([0.1, 0.5, 1.0, 3.0])

    expected, _ = theirs(inputs, state, dt[:, None])
    actual = ours(inputs, state, dt=dt)
    torch.testing.assert_close(actual, expected)

    expected.sum().backward()
    actual.sum().backward()
    their_grads = dict(theirs.named_parameters())
    for name, parameter in ours.named_parameters():
        torch.testing.assert_close(parameter.grad, their_grads[name].grad)
