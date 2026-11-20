import torch

from .cell import CfcCell


def run_cfc_sequence(
    cell: CfcCell,
    inputs: torch.Tensor,
    *,
    dt,
    initial_state: torch.Tensor | None = None,
    collect_history: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Esegue la CfC su [batch, istanti, ingressi]; restituisce storia opzionale e stato finale.

    dt: numero (passo fisso) oppure [batch, istanti], cioè il tempo trascorso prima
    di ciascun campione: con eventi irregolari ogni passo ha il proprio Δt.
    """
    if inputs.ndim != 3:
        raise ValueError("inputs deve avere forma [batch, istanti, ingressi].")
    batch, steps, _ = inputs.shape
    if batch == 0 or steps == 0:
        raise ValueError("Batch e sequenza non possono essere vuoti.")

    dt = torch.as_tensor(dt, dtype=inputs.dtype, device=inputs.device)
    if dt.ndim == 0:
        dt = dt.expand(batch, steps)
    if dt.shape != (batch, steps):
        raise ValueError("dt deve essere scalare o avere forma [batch, istanti].")

    state = cell.initial_state(batch) if initial_state is None else initial_state
    history = [] if collect_history else None

    for index in range(steps):
        state = cell(inputs[:, index, :], state, dt=dt[:, index])
        if history is not None:
            history.append(state)

    return (torch.stack(history, dim=1) if history is not None else None), state
