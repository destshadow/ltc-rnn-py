import torch

from .neuron_parameters import NeuronParameters
from .synapse_parameters import SynapseParameters
from .synapses import compute_synaptic_effects
from .validation import validate_duration, validate_solver_inputs


def semi_implicit_step(
    state: torch.Tensor,
    neurons: NeuronParameters,
    drive: torch.Tensor,
    total_conductance: torch.Tensor,
    step_duration: float,
) -> torch.Tensor:
    """Calcola un singolo piccolo aggiornamento dello stato."""
    validate_duration(step_duration)

    capacity_rate = neurons.capacitance / step_duration
    leak = neurons.leak_conductance

    numerator = (
        capacity_rate * state
        + leak * neurons.resting_potential
        + drive
    )

    denominator = capacity_rate + leak + total_conductance #conduttanze, influenzano sia la velocità di risposta sia la direzione verso cui viene spinto lo stato

    return numerator / denominator


def advance_state(
    inputs: torch.Tensor,
    state: torch.Tensor,
    neurons: NeuronParameters,
    sensory: SynapseParameters,
    recurrent: SynapseParameters,
    *,
    dt: float,
    substeps: int,
) -> torch.Tensor:
    """Fa avanzare lo stato per un intervallo dt."""
    validate_duration(dt)
    validate_solver_inputs(inputs, state, neurons, sensory, recurrent)

    if type(substeps) is not int or substeps <= 0:
        raise ValueError("substeps deve essere un intero positivo.")

    step_duration = dt / substeps

    sensory_drive, sensory_conductance = compute_synaptic_effects(
        inputs, sensory
    )

    for _ in range(substeps):
        recurrent_drive, recurrent_conductance = compute_synaptic_effects(
            state, recurrent
        )

        state = semi_implicit_step(
            state=state,
            neurons=neurons,
            drive=sensory_drive + recurrent_drive,
            total_conductance=(
                sensory_conductance + recurrent_conductance
            ),
            step_duration=step_duration,
        )

    return state
