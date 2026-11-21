import torch
from torch import nn


class MultiOutputLoss(nn.Module):
    """MSE pesata sulle uscite (obiettivi standardizzati, uno per colonna).

    I pesi decidono quanto conta ogni stima (es. volatilità più del drift); sono
    normalizzati a somma 1, così la scala della perdita non dipende dal numero di uscite.
    """

    def __init__(self, outputs: tuple[str, ...], weights: dict[str, float] | None = None):
        super().__init__()
        weights = weights or {name: 1.0 for name in outputs}
        if set(weights) != set(outputs):
            raise ValueError("Serve un peso per ogni uscita, e solo per quelle.")
        values = torch.tensor([float(weights[name]) for name in outputs])
        if not torch.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("I pesi devono essere finiti, non negativi e non tutti nulli.")
        self.outputs = tuple(outputs)
        self.register_buffer("weights", values / values.sum())

    def per_output(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """MSE di ciascuna uscita, per i report."""
        errors = self._squared_errors(predictions, targets).mean(dim=0)
        return {name: errors[index].item() for index, name in enumerate(self.outputs)}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # I pesi seguono il dispositivo delle previsioni (training su GPU, validazione su CPU).
        weights = self.weights.to(device=predictions.device, dtype=predictions.dtype)
        return (self._squared_errors(predictions, targets).mean(dim=0) * weights).sum()

    def _squared_errors(self, predictions, targets):
        if predictions.shape != targets.shape or predictions.ndim != 2 \
                or predictions.shape[1] != len(self.outputs):
            raise ValueError("predictions e targets devono avere forma [batch, uscite].")
        return (predictions - targets) ** 2
