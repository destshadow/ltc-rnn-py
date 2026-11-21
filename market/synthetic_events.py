import math

import torch


def regime_events(
    count: int,
    *,
    drifts: tuple[float, ...] = (-0.5, 0.0, 0.5),
    volatilities: tuple[float, ...] = (0.5, 1.0, 2.0),
    mean_dt: float = 1.0,
    switch_probability: float = 0.01,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Eventi irregolari con drift e volatilità noti che cambiano regime.

    Δt ~ esponenziale (media mean_dt); a ogni evento, con probabilità switch_probability,
    drift e volatilità vengono riestratti fra i valori indicati. Incremento del log-prezzo:
    r = μ·Δt + σ·√Δt·ε, ε ~ N(0, 1). Restituisce dt, returns, true_drift, true_volatility
    (i valori veri in vigore durante ciascun evento), tutti [count] float64.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count deve essere un intero positivo.")
    if not mean_dt > 0 or not 0 <= switch_probability <= 1:
        raise ValueError("mean_dt deve essere positivo e switch_probability in [0, 1].")
    if not drifts or not volatilities or min(volatilities) <= 0:
        raise ValueError("Servono drift e volatilità, con volatilità positive.")

    generator = torch.Generator().manual_seed(seed)
    dt = torch.empty(count, dtype=torch.float64).exponential_(1 / mean_dt, generator=generator)
    noise = torch.randn(count, generator=generator, dtype=torch.float64)
    switches = torch.rand(count, generator=generator, dtype=torch.float64) < switch_probability
    drift_choice = torch.randint(len(drifts), (count,), generator=generator)
    volatility_choice = torch.randint(len(volatilities), (count,), generator=generator)

    true_drift = torch.empty(count, dtype=torch.float64)
    true_volatility = torch.empty(count, dtype=torch.float64)
    drift, volatility = drifts[int(drift_choice[0])], volatilities[int(volatility_choice[0])]
    for index in range(count):
        if switches[index]:
            drift = drifts[int(drift_choice[index])]
            volatility = volatilities[int(volatility_choice[index])]
        true_drift[index], true_volatility[index] = drift, volatility

    returns = true_drift * dt + true_volatility * torch.sqrt(dt) * noise
    return {"dt": dt, "returns": returns, "true_drift": true_drift,
            "true_volatility": true_volatility}


def regime_statistics(events: dict[str, torch.Tensor]) -> dict[tuple[float, float], dict]:
    """Per ogni coppia (drift, volatilità) vera: drift e volatilità stimati dagli eventi."""
    keys = torch.stack([events["true_drift"], events["true_volatility"]], dim=1)
    statistics = {}
    for key in torch.unique(keys, dim=0):
        mask = (keys == key).all(dim=1)
        dt, returns = events["dt"][mask], events["returns"][mask]
        standardized = (returns - key[0] * dt) / torch.sqrt(dt)
        statistics[(key[0].item(), key[1].item())] = {
            "count": int(mask.sum()),
            "drift": (returns.sum() / dt.sum()).item(),
            "volatility": standardized.std().item() if mask.sum() > 1 else math.nan,
        }
    return statistics
