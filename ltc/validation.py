import math
from numbers import Real

import torch
from torch import nn


def validate_duration(duration: float) -> None:
    """Verifica una durata scalare, positiva e finita."""
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise TypeError("La durata deve essere un numero reale scalare.")

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("La durata deve essere finita e maggiore di zero.")


def validate_solver_inputs(
    inputs: torch.Tensor,
    state: torch.Tensor,
    neurons: nn.Module,
    sensory: nn.Module,
    recurrent: nn.Module,
) -> None:
    """Verifica forme, dispositivi e tipi numerici del solver."""
    for name, tensor in (("inputs", inputs), ("state", state)):
        if tensor.ndim != 2:
            raise ValueError(f"{name} deve avere due dimensioni.")

        if any(size <= 0 for size in tensor.shape):
            raise ValueError(f"{name} non può avere dimensioni vuote.")

        if not tensor.is_floating_point():
            raise TypeError(f"{name} deve usare numeri in virgola mobile.")

    if inputs.shape[0] != state.shape[0]:
        raise ValueError("inputs e state devono avere lo stesso batch_size.")

    if inputs.device != state.device:
        raise ValueError("inputs e state devono essere sullo stesso dispositivo.")

    if inputs.dtype != state.dtype:
        raise TypeError("inputs e state devono avere lo stesso tipo numerico.")

    input_size = inputs.shape[1]
    hidden_size = state.shape[1]

    groups = (
        ("neurons", neurons, (hidden_size,)),
        ("sensory", sensory, (input_size, hidden_size)),
        ("recurrent", recurrent, (hidden_size, hidden_size)),
    )

    for group_name, module, expected_shape in groups:
        for parameter_name, parameter in module.named_parameters():
            name = f"{group_name}.{parameter_name}"

            if tuple(parameter.shape) != expected_shape:
                raise ValueError(
                    f"{name}: forma {tuple(parameter.shape)}, "
                    f"attesa {expected_shape}."
                )

            if parameter.device != state.device:
                raise ValueError(
                    f"{name} deve essere sullo stesso dispositivo dello stato."
                )

            if parameter.dtype != state.dtype:
                raise TypeError(
                    f"{name} deve avere lo stesso tipo numerico dello stato."
                )