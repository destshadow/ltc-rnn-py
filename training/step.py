import torch
from torch import nn


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    dt: float,
) -> float:
    """Esegue un aggiornamento dei parametri su un gruppo di esempi."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(inputs, dt=dt)
    loss = loss_function(logits, labels)

    if not torch.isfinite(loss).item():
        raise RuntimeError("L'errore di addestramento non è finito.")

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )

    optimizer.step()

    return loss.item()