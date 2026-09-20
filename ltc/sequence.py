import torch

from .cell import LTCCell


def run_sequence(
    cell: LTCCell,
    inputs: torch.Tensor,
    *,
    dt: float,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restituisce la storia degli stati e lo stato finale."""
    if inputs.ndim != 3:
        raise ValueError(
            "inputs deve avere forma [batch, istanti, ingressi]."
        )

    batch_size, sequence_length, input_size = inputs.shape

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Batch e sequenza non possono essere vuoti.")

    if input_size != cell.config.input_size:
        raise ValueError("Il numero di ingressi non coincide con la cella.")

    if initial_state is None:
        state = cell.initial_state(batch_size)
    else:
        state = initial_state

    history = []

    for index in range(sequence_length):
        current_inputs = inputs[:, index, :]
        state = cell(current_inputs, state, dt=dt)
        history.append(state)

    return torch.stack(history, dim=1), state