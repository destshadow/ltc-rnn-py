import torch


def return_stats(returns: torch.Tensor) -> dict:
    """Statistiche di qualità di una serie di rendimenti, per confrontare periodi.

    lag1_autocorrelation: quanto il rendimento di oggi somiglia a quello di ieri
    (positiva = inerzia, negativa = ritorno verso la media).
    extreme_share: frazione di giorni oltre 4 deviazioni standard del periodo stesso.
    """
    if returns.ndim != 1 or len(returns) < 3:
        raise ValueError("Servono almeno tre rendimenti [n].")

    values = returns.double()
    std = values.std()
    if not std > 0:
        raise ValueError("Rendimenti costanti: statistiche non definite.")

    centered = values - values.mean()
    autocorrelation = (centered[:-1] * centered[1:]).sum() / centered.pow(2).sum()

    return {
        "count": len(values),
        "mean": values.mean().item(),
        "volatility": std.item(),
        "lag1_autocorrelation": autocorrelation.item(),
        "max_abs": values.abs().max().item(),
        "extreme_share": ((centered.abs() / std) > 4).double().mean().item(),
    }
