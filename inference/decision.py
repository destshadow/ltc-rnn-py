import math

import torch


def classify_binary(
    logits: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Converte punteggi [batch, 2] nelle classi 0 oppure 1."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("logits deve avere forma [batch, 2].")

    if not logits.is_floating_point():
        raise TypeError("I punteggi devono essere in virgola mobile.")

    if not math.isfinite(threshold):
        raise ValueError("La soglia deve essere finita.")

    if not torch.isfinite(logits).all().item():
        raise ValueError("I punteggi devono essere finiti.")

    margins = (logits[:, 0] - logits[:, 1]).double()

    return (margins < threshold).long()