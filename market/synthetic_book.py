import torch

from .synthetic_events import regime_events


def regime_path(count: int, values: tuple[float, ...], switch_probability: float,
                generator: torch.Generator) -> torch.Tensor:
    """Valore in vigore a ogni evento: con probabilità switch_probability viene riestratto."""
    switches = torch.rand(count, generator=generator) < switch_probability
    choices = torch.randint(len(values), (count,), generator=generator)
    path = torch.empty(count, dtype=torch.float64)
    current = values[int(choices[0])]
    for index in range(count):
        if switches[index]:
            current = values[int(choices[index])]
        path[index] = current
    return path


def book_events(
    count: int,
    *,
    toxicities: tuple[float, ...] = (0.1, 0.3, 0.6),
    ks: tuple[float, ...] = (0.5, 1.0, 2.0),
    switch_probability: float = 0.01,
    seed: int = 0,
    **price_options,
) -> dict[str, torch.Tensor]:
    """Eventi di regime_events con un ordine a mercato ciascuno: lato e portata.

    side ∈ {−1, +1}: con probabilità pari alla tossicità vera l'ordine è INFORMATO e segue il
    segno del rendimento dell'evento successivo; altrimenti il lato è casuale.
    reach ≥ 0 (in unità di prezzo dal mid): esponenziale di parametro k vero, quindi
    P(reach ≥ δ) = exp(−k·δ) (forma di Avellaneda-Stoikov). È la probabilità di esecuzione
    di un nostro ordine limite a distanza δ CONDIZIONATA a un ordine a mercato del lato
    compatibile (acquisto per un nostro ask, vendita per un bid); per evento va moltiplicata
    per la probabilità di quel lato. Qui si genera il parametro della portata, non un
    simulatore di esecuzioni (code e intensità temporali non sono modellate).
    Tossicità e k cambiano per regimi indipendenti da quelli del prezzo.
    """
    if min(toxicities, default=-1) < 0 or max(toxicities, default=2) > 1:
        raise ValueError("Le tossicità devono essere probabilità in [0, 1].")
    if not ks or min(ks) <= 0:
        raise ValueError("Servono valori di k positivi.")

    events = regime_events(count, switch_probability=switch_probability, seed=seed, **price_options)
    generator = torch.Generator().manual_seed(seed + 1_000_003)   # flusso diverso dal prezzo
    true_toxicity = regime_path(count, toxicities, switch_probability, generator)
    true_k = regime_path(count, ks, switch_probability, generator)

    informed = torch.rand(count, generator=generator, dtype=torch.float64) < true_toxicity
    next_returns = torch.cat([events["returns"][1:], events["returns"][-1:]])
    future_sign = torch.where(next_returns >= 0, 1.0, -1.0).double()
    random_side = (torch.randint(2, (count,), generator=generator) * 2 - 1).double()
    reach = torch.empty(count, dtype=torch.float64).exponential_(1.0, generator=generator) / true_k

    return {**events, "side": torch.where(informed, future_sign, random_side), "reach": reach,
            "informed": informed, "true_toxicity": true_toxicity, "true_k": true_k}
