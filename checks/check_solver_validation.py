import torch

from ltc.config import LTCConfig
from ltc.neuron_parameters import NeuronParameters
from ltc.solver import advance_state, semi_implicit_step
from ltc.synapse_parameters import SynapseParameters


def expect_error(label, error_type, action):
    try:
        action()
    except error_type as error:
        print(f"{label}: rifiutato — {error}")
    else:
        raise AssertionError(f"{label}: il solver doveva rifiutare il caso.")


def main():
    # Questi controlli funzionano anche senza una GPU.
    config = LTCConfig(input_size=3, hidden_size=8)
    neurons = NeuronParameters(config)
    sensory = SynapseParameters(3, 8)
    recurrent = SynapseParameters(8, 8)

    inputs = torch.zeros(2, 3)
    state = torch.zeros(2, 8)

    def run(candidate_inputs, candidate_state):
        return advance_state(
            candidate_inputs,
            candidate_state,
            neurons,
            sensory,
            recurrent,
            dt=0.1,
            substeps=config.substeps,
        )

    with torch.no_grad():
        result = run(inputs, state)

    assert tuple(result.shape) == (2, 8)
    assert torch.isfinite(result).all().item()
    print("Caso valido: accettato.")

    expect_error(
        "Batch diversi",
        ValueError,
        lambda: run(torch.zeros(1, 3), state),
    )

    expect_error(
        "Numero di ingressi sbagliato",
        ValueError,
        lambda: run(torch.zeros(2, 4), state),
    )

    expect_error(
        "Numero di neuroni sbagliato",
        ValueError,
        lambda: run(inputs, torch.zeros(2, 7)),
    )

    expect_error(
        "Tipi numerici diversi",
        TypeError,
        lambda: run(inputs.double(), state),
    )

    expect_error(
        "Durata nulla nel passo elementare",
        ValueError,
        lambda: semi_implicit_step(
            state, neurons, torch.zeros_like(state),
            torch.zeros_like(state), step_duration=0.0,
        ),
    )

    print("Controlli superati.")


if __name__ == "__main__":
    main()