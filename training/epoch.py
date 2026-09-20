import torch
from torch import nn

from .step import train_step


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    dt: float,
    batch_size: int = 64,
) -> float:
    """Esegue un passaggio su tutti gli esempi, in ordine casuale."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size deve essere un intero positivo.")

    if len(inputs) == 0 or labels.shape != (len(inputs),):
        raise ValueError("Servono dati non vuoti e un'etichetta per esempio.")

    if inputs.device != labels.device:
        raise ValueError("Ingressi ed etichette devono avere lo stesso dispositivo.")

    order = torch.randperm(len(inputs), device=inputs.device)
    total_loss = 0.0

    for start in range(0, len(inputs), batch_size):
        indices = order[start:start + batch_size]

        loss = train_step(
            model,
            optimizer,
            loss_function,
            inputs[indices],
            labels[indices],
            dt=dt,
        )

        total_loss += loss * len(indices)

    return total_loss / len(inputs)