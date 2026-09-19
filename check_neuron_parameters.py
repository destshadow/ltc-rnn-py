import torch

from ltc.config import LTCConfig
from ltc.neuron_parameters import NeuronParameters


def main():
    config = LTCConfig(input_size=3, hidden_size=8)
    params = NeuronParameters(config).to("cuda")

    total = sum(parameter.numel() for parameter in params.parameters())
    print("Numeri apprendibili:", total)
    assert total == 24

    for name, parameter in params.named_parameters():
        print(
            name,
            "forma:", tuple(parameter.shape),
            "dispositivo:", parameter.device,
        )

    assert torch.all(params.capacitance > 0).item()
    assert torch.all(params.leak_conductance > 0).item()

    print("Capacità iniziale:", round(params.capacitance[0].item(), 6))
    print("Ritorno al riposo:", round(params.leak_conductance[0].item(), 6))

    # Un calcolo artificiale per verificare il percorso dei gradienti.
    probe = (
        params.capacitance.sum()
        + params.leak_conductance.sum()
        + params.resting_potential.sum()
    )

    probe.backward()

    for name, parameter in params.named_parameters():
        assert parameter.grad is not None, f"Gradiente assente: {name}"
        assert torch.isfinite(parameter.grad).all().item()
        assert torch.count_nonzero(parameter.grad).item() > 0

    print("Gradienti presenti, finiti e non nulli.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()