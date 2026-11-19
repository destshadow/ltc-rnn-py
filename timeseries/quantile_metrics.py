import torch


def _check(quantiles: torch.Tensor, levels, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("targets deve avere forma [n] non vuota.")
    if quantiles.shape != (len(targets), len(levels)):
        raise ValueError("quantiles deve avere forma [n, livelli].")
    if any(not 0 < level < 1 for level in levels):
        raise ValueError("I livelli devono essere compresi tra 0 e 1.")
    return torch.tensor(levels, dtype=torch.float64)


def pinball_loss(quantiles: torch.Tensor, levels, targets: torch.Tensor) -> float:
    """Perdita quantile media su esempi e livelli: più bassa è meglio.

    Per il livello q: q·(y − ŷ) se y ≥ ŷ, altrimenti (1 − q)·(ŷ − y).
    """
    tau = _check(quantiles, levels, targets)
    error = targets.double()[:, None] - quantiles.double()
    return torch.maximum(tau * error, (tau - 1) * error).mean().item()


def interval_coverage(
    quantiles: torch.Tensor,
    levels,
    targets: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> dict:
    """Frazione di obiettivi dentro [quantile lower, quantile upper] e larghezza media.

    Con lower=0.1 e upper=0.9 un modello ben tarato copre circa l'80%.
    """
    _check(quantiles, levels, targets)
    levels = list(levels)
    if lower not in levels or upper not in levels or lower >= upper:
        raise ValueError("lower e upper devono essere livelli presenti, con lower < upper.")

    low = quantiles[:, levels.index(lower)].double()
    high = quantiles[:, levels.index(upper)].double()
    inside = (targets.double() >= low) & (targets.double() <= high)

    return {
        "nominal": upper - lower,
        "coverage": inside.double().mean().item(),
        "mean_width": (high - low).mean().item(),
    }
