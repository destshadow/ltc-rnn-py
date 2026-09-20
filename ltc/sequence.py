import torch

from .cell import LTCCell


def run_sequence(
    cell: LTCCell,
    inputs: torch.Tensor,
    *,
    dt: float,
    initial_state: torch.Tensor | None = None,
    collect_history: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Restituisce la storia opzionale e lo stato finale."""
    if inputs.ndim != 3:
        raise ValueError(
            "inputs deve avere forma [batch, istanti, ingressi]."
        )

    batch_size, sequence_length, input_size = inputs.shape

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("Batch e sequenza non possono essere vuoti.")

    if input_size != cell.config.input_size:
        raise ValueError("Il numero di ingressi non coincide con la cella.")

    if type(collect_history) is not bool:
        raise TypeError("collect_history deve essere un booleano.")

    if initial_state is None:
        state = cell.initial_state(batch_size)
    else:
        state = initial_state

    history = [] if collect_history else None

    for index in range(sequence_length):
        state = cell(inputs[:, index, :], state, dt=dt)

        if history is not None:
            history.append(state)

    stacked_history = (
        torch.stack(history, dim=1)
        if history is not None
        else None
    )

    return stacked_history, state