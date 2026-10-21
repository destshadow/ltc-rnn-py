import torch

from ltc.config import LTCConfig
from ltc.neuron_parameters import NeuronParameters
from ltc.solver import semi_implicit_step
from ltc.state import create_initial_state


def simulate_decay(neurons, config, substeps):
    state = create_initial_state(
        config, batch_size=1, device="cuda"
    )
    state.fill_(1.0)

    no_connections = torch.zeros_like(state)
    step_duration = 1.0 / substeps

    for _ in range(substeps):
        previous = state

        state = semi_implicit_step(
            state=state,
            neurons=neurons,
            drive=no_connections,
            total_conductance=no_connections,
            step_duration=step_duration,
        )

        assert torch.all(state > 0).item()
        assert torch.all(state < previous).item()

    return state


def main():
    config = LTCConfig(input_size=1, hidden_size=1)
    neurons = NeuronParameters(config).to("cuda")

    with torch.no_grad():
        coarse = simulate_decay(neurons, config, substeps=1)
        fine = simulate_decay(neurons, config, substeps=100)

        # Soluzione esatta di questo specifico caso semplice,
        # dopo un intervallo di durata 1.
        exact = torch.exp(
            -neurons.leak_conductance / neurons.capacitance
        ).unsqueeze(0)

        coarse_error = (coarse - exact).abs().item()
        fine_error = (fine - exact).abs().item()

    print(f"Un solo passo: {coarse.item():.6f}")
    print(f"Cento piccoli passi: {fine.item():.6f}")
    print(f"Riferimento esatto: {exact.item():.6f}")

    assert fine_error < coarse_error
    assert fine_error < 0.002

    print("Decadimento corretto.")
    print("Con più piccoli passi il risultato è più accurato.")
    print("Controlli superati.")


if __name__ == "__main__":
    main()