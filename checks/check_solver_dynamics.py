import math

import torch

from ltc.config import LTCConfig
from ltc.neuron_parameters import NeuronParameters
from ltc.solver import advance_state
from ltc.synapse_parameters import SynapseParameters


def create_components():
    config = LTCConfig(input_size=1, hidden_size=1)

    neurons = NeuronParameters(config).double()
    sensory = SynapseParameters(1, 1).double()
    recurrent = SynapseParameters(1, 1).double()

    with torch.no_grad():
        neurons.raw_capacitance.zero_()
        neurons.raw_leak_conductance.zero_()
        neurons.resting_potential.zero_()

        for connections in (sensory, recurrent):
            connections.raw_strength.zero_()
            connections.raw_slope.zero_()
            connections.threshold.zero_()

        sensory.reversal_potential.fill_(1.0)
        recurrent.reversal_potential.fill_(-0.5)

    return neurons, sensory, recurrent

def check_dynamics():
    neurons, sensory, recurrent = create_components()

    inputs = torch.zeros(1, 1, dtype=torch.float64)
    initial_state = torch.zeros_like(inputs)

    with torch.no_grad():
        first = advance_state(
            inputs, initial_state, neurons, sensory, recurrent,
            dt=1.0, substeps=1,
        )

        second = advance_state(
            inputs, first, neurons, sensory, recurrent,
            dt=1.0, substeps=1,
        )

        combined = advance_state(
            inputs, initial_state, neurons, sensory, recurrent,
            dt=2.0, substeps=2,
        )

        # Riferimenti scalari ricavati per questo caso controllato.
        expected_first = 1.0 / 12.0

        slope = recurrent.slope.item()
        recurrent_opening = 1.0 / (
            1.0 + math.exp(-slope * expected_first)
        )

        expected_second = (
            expected_first + 0.5 - 0.5 * recurrent_opening
        ) / (2.5 + recurrent_opening)

    torch.testing.assert_close(
        first,
        torch.full_like(first, expected_first),
        rtol=1e-10,
        atol=1e-12,
    )

    torch.testing.assert_close(
        second,
        torch.full_like(second, expected_second),
        rtol=1e-10,
        atol=1e-12,
    )

    torch.testing.assert_close(
        combined, second, rtol=1e-10, atol=1e-12
    )

    torch.testing.assert_close(
        initial_state, torch.zeros_like(initial_state)
    )

    print(f"Primo stato: {first.item():.6f}")
    print("Secondo stato: riferimento verificato.")
    print("Due chiamate e due sottopassi: risultati equivalenti.")
    print("Stato iniziale: non modificato.")

def check_temporal_gradients():
    neurons, sensory, recurrent = create_components()
    state = torch.zeros(1, 1, dtype=torch.float64)

    sequence = [
        torch.tensor(
            [[value]], dtype=torch.float64, requires_grad=True
        )
        for value in (0.2, -0.1, 0.3)
    ]

    for inputs in sequence:
        state = advance_state(
            inputs, state, neurons, sensory, recurrent,
            dt=0.1, substeps=4,
        )

    state.sum().backward()

    for index, inputs in enumerate(sequence):
        assert inputs.grad is not None, f"Gradiente assente: ingresso {index}"
        assert torch.isfinite(inputs.grad).all().item()
        assert torch.count_nonzero(inputs.grad).item() > 0

    for module in (neurons, sensory, recurrent):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"Gradiente assente: {name}"
            assert torch.isfinite(parameter.grad).all().item()

    print("Lo stato finale dipende da tutti e tre gli ingressi.")
    print("Gradienti dei parametri: presenti e finiti.")

def main():
    check_dynamics()
    check_temporal_gradients()
    print("Controlli superati.")


if __name__ == "__main__":
    main()