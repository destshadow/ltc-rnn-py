import torch


def return_stats(runs: list[torch.Tensor]) -> dict:
    """Statistiche di qualità di rendimenti divisi in tratti consecutivi, per confrontare periodi.

    runs: un tensore [n] per tratto; fra un tratto e l'altro c'è un'interruzione,
    quindi le coppie lag-1 si formano solo DENTRO ciascun tratto.
    lag1_autocorrelation: Σ (r_t − m)(r_{t+1} − m) sulle coppie interne, diviso
    Σ (r_t − m)² su tutti i rendimenti (m = media complessiva). Positiva = inerzia,
    negativa = ritorno verso la media.
    extreme_share: frazione di giorni oltre 4 deviazioni standard del periodo stesso.
    """
    if not runs or any(run.ndim != 1 for run in runs):
        raise ValueError("runs deve essere una lista non vuota di tensori [n].")

    values = torch.cat(runs).double()
    if len(values) < 3:
        raise ValueError("Servono almeno tre rendimenti.")

    mean = values.mean()
    std = values.std()
    if not std > 0:
        raise ValueError("Rendimenti costanti: statistiche non definite.")

    pairs = sum(
        ((run.double()[:-1] - mean) * (run.double()[1:] - mean)).sum()
        for run in runs if len(run) > 1
    )
    centered = values - mean

    return {
        "count": len(values),
        "mean": mean.item(),
        "volatility": std.item(),
        "lag1_autocorrelation": (pairs / centered.pow(2).sum()).item(),
        "max_abs": values.abs().max().item(),
        "extreme_share": ((centered.abs() / std) > 4).double().mean().item(),
    }
