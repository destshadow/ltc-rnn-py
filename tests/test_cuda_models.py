"""CfC and multi-output model on the GPU give the CPU results (skipped without CUDA)."""

import pytest
import torch

from cfc.config import CfcConfig
from timeseries.model import MultiOutputForecaster
from training.epoch import train_epoch
from training.multi_loss import MultiOutputLoss

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA non disponibile")]
OUTPUTS = ("drift", "volatility")


def batch():
    torch.manual_seed(1)
    return torch.stack([torch.randn(8, 6), torch.rand(8, 6) + 0.5], dim=2)


def test_same_forward_on_cpu_and_gpu():
    torch.manual_seed(0)
    model = MultiOutputForecaster(CfcConfig(2, 5, backbone_units=8), OUTPUTS, elapsed_channel=1)
    inputs = batch()
    expected = model(inputs, dt=0.3)
    actual = model.to("cuda")(inputs.to("cuda"), dt=0.3).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_multi_output_training_step_runs_on_gpu():
    torch.manual_seed(0)
    model = MultiOutputForecaster(CfcConfig(2, 5, backbone_units=8), OUTPUTS,
                                  elapsed_channel=1).to("cuda")
    loss = MultiOutputLoss(OUTPUTS, {"drift": 1.0, "volatility": 2.0})   # weights stay on CPU
    inputs, targets = batch().to("cuda"), torch.randn(8, 2, device="cuda")
    value = train_epoch(model, torch.optim.Adam(model.parameters()), loss, inputs, targets,
                        dt=0.1, batch_size=4)
    assert value > 0
    assert loss(torch.zeros(3, 2), torch.ones(3, 2)).item() == pytest.approx(1.0)   # CPU still works


def test_same_seed_gives_the_same_training_on_cpu_and_gpu():
    def one_epoch(device):
        torch.manual_seed(0)
        model = MultiOutputForecaster(CfcConfig(2, 5, backbone_units=8), OUTPUTS,
                                      elapsed_channel=1).to(device)
        inputs, targets = batch().to(device), torch.randn(8, 2).to(device)
        torch.manual_seed(42)                       # the batch order comes from here
        train_epoch(model, torch.optim.SGD(model.parameters(), lr=0.1), MultiOutputLoss(OUTPUTS),
                    inputs, targets, dt=0.1, batch_size=2)
        return {name: value.detach().cpu() for name, value in model.state_dict().items()}

    cpu, gpu = one_epoch("cpu"), one_epoch("cuda")
    for name in cpu:   # same batch order → same updates, up to float rounding
        torch.testing.assert_close(gpu[name], cpu[name], rtol=1e-4, atol=1e-5)
