
from .validation import validate_duration, validate_solver_inputs
import torch

from .neuron_parameters import NeuronParameters
from .synapse_parameters import SynapseParameters
from .synapses import compute_synaptic_effects


def semi_implicit_step(
    state: torch.Tensor,
    neurons: NeuronParameters,
    drive: torch.Tensor,
    total_conductance: torch.Tensor,
    step_duration: float,
) -> torch.Tensor:
    """Calcola un singolo piccolo aggiornamento dello stato."""
    capacity_rate = neurons.capacitance / step_duration
    leak = neurons.leak_conductance

    numerator = (
        capacity_rate * state
        + leak * neurons.resting_potential
        + drive
    )

    denominator = capacity_rate + leak + total_conductance

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
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt deve essere finito e maggiore di zero.")

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