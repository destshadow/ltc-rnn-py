import torch
from torch import nn


@torch.no_grad()
def predict(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    dt: float,
    batch_size: int = 256,
) -> torch.Tensor:
    """Previsioni [n] sulla CPU, senza aggiornare i parametri."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size deve essere un intero positivo.")

    model.eval()
    device = next(model.parameters()).device
    outputs = []

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size].to(device)
        prediction = model(batch, dt=dt)

        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("Il modello ha prodotto previsioni non finite.")

        outputs.append(prediction.cpu())

    return torch.cat(outputs)


def regression_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    """Errori e frazione di segni indovinati.

    Il segno conta solo dove obiettivo e previsione sono diversi da zero:
    una previsione nulla non sceglie una direzione (NaN se non ne sceglie mai).
    """
    if predictions.shape != targets.shape or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("Servono previsioni e obiettivi [n] della stessa forma.")

    errors = predictions.double() - targets.double()
    decided = (targets != 0) & (predictions != 0)
    hits = (torch.sign(predictions) == torch.sign(targets)) & decided

    return {
        "count": len(targets),
        "mse": errors.pow(2).mean().item(),
        "mae": errors.abs().mean().item(),
        "direction_accuracy": (
            hits.sum().item() / decided.sum().item()
            if decided.any().item()
            else float("nan")
        ),
    }


def baseline_predictions(
    targets: torch.Tensor,
    *,
    training_mean: float,
) -> dict[str, torch.Tensor]:
    """Riferimenti da battere, con la stessa forma degli obiettivi.

    zero: rendimento nullo, cioè "il prezzo resta quello di oggi".
    training_mean: rendimento medio del training, costante.
    """
    return {
        "zero": torch.zeros_like(targets),
        "training_mean": torch.full_like(targets, training_mean),
    }
