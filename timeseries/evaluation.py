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
    """Errori, frazione di segni indovinati e diagnosi dell'errore.

    Il segno conta solo dove obiettivo e previsione sono diversi da zero:
    una previsione nulla non sceglie una direzione. Se non ne sceglie mai,
    la metrica è None (null in JSON), non un numero.
    Diagnosi: mse = level_error + dynamic_error; spread_ratio = oscillazione delle
    previsioni rispetto agli obiettivi; correlation None se una delle due è costante.
    """
    if predictions.shape != targets.shape or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("Servono previsioni e obiettivi [n] della stessa forma.")

    errors = predictions.double() - targets.double()
    decided = (targets != 0) & (predictions != 0)
    hits = (torch.sign(predictions) == torch.sign(targets)) & decided

    # MSE = livello + dinamica: (media previsioni − media obiettivi)² + varianza degli scarti.
    level_error = errors.mean().pow(2).item()
    prediction_std = predictions.double().std(unbiased=False).item()
    target_std = targets.double().std(unbiased=False).item()
    correlation = (
        torch.corrcoef(torch.stack([predictions.double(), targets.double()]))[0, 1].item()
        if prediction_std > 0 and target_std > 0 else None
    )

    return {
        "count": len(targets),
        "mse": errors.pow(2).mean().item(),
        "mae": errors.abs().mean().item(),
        "level_error": level_error,
        "dynamic_error": errors.var(unbiased=False).item(),
        # Oscillazione relativa: vicino a 0 = previsione quasi costante.
        "spread_ratio": prediction_std / target_std if target_std > 0 else None,
        "correlation": correlation,
        "direction_accuracy": (
            hits.sum().item() / decided.sum().item()
            if decided.any().item()
            else None
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


def tail_split_mse(predictions: torch.Tensor, targets: torch.Tensor, *, quantile: float = 0.95) -> dict:
    """MSE separato fra i giorni con obiettivo più alto (coda) e tutti gli altri.

    Mostra se un errore complessivo viene da pochi giorni estremi (es. crisi).
    """
    if predictions.shape != targets.shape or targets.ndim != 1 or len(targets) < 2:
        raise ValueError("Servono previsioni e obiettivi [n] della stessa forma, n >= 2.")
    if not 0 < quantile < 1:
        raise ValueError("quantile deve essere compreso tra 0 e 1.")

    threshold = torch.quantile(targets.double(), quantile)
    tail = targets.double() >= threshold
    squared = (predictions.double() - targets.double()) ** 2
    return {
        "quantile": quantile,
        "tail_count": int(tail.sum()),
        "tail_mse": squared[tail].mean().item(),
        "rest_mse": squared[~tail].mean().item() if (~tail).any() else None,
    }
