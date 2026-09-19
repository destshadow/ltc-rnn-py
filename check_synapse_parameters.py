import torch

from ltc.config import LTCConfig
from ltc.synapse_parameters import SynapseParameters


def check_group(name, connections, expected_shape):
    total = sum(p.numel() for p in connections.parameters())

    print(f"{name}: forma {expected_shape}, numeri apprendibili {total}")

    assert total == 4 * expected_shape[0] * expected_shape[1]

    for parameter in connections.parameters():
        assert tuple(parameter.shape) == expected_shape
        assert parameter.device.type == "cuda"
        assert torch.isfinite(parameter).all().item()

    assert torch.all(connections.strength > 0).item()
    assert torch.all(connections.slope > 0).item()

    probe = (
        connections.strength.sum()
        + connections.slope.sum()
        + connections.threshold.sum()
        + connections.reversal_potential.sum()
    )

    probe.backward()

    for parameter in connections.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()
        assert torch.count_nonzero(parameter.grad).item() > 0


def main():
    torch.manual_seed(42)

    config = LTCConfig(input_size=3, hidden_size=8)

    sensory = SynapseParameters(
        source_size=config.input_size,
        target_size=config.hidden_size,
    ).to("cuda")

    recurrent = SynapseParameters(
        source_size=config.hidden_size,
        target_size=config.hidden_size,
    ).to("cuda")

    check_group("Ingressi", sensory, (3, 8))
    check_group("Ricorrenti", recurrent, (8, 8))

    print("Positività e gradienti verificati.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()