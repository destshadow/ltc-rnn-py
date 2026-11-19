from datetime import datetime, timedelta

import torch


def ar1_prices(
    count: int,
    *,
    phi: float,
    sigma: float = 0.01,
    seed: int = 0,
    start: datetime = datetime(2000, 1, 1),
) -> tuple[list[datetime], torch.Tensor]:
    """Prezzi giornalieri continui con rendimenti AR(1): r_t = phi·r_{t-1} + sigma·e_t.

    Serve da controllo: con phi ≠ 0 una parte nota dei rendimenti è prevedibile
    (frazione phi² della varianza); con phi = 0 non lo è affatto.
    """
    if type(count) is not int or count < 2:
        raise ValueError("count deve essere un intero >= 2.")
    if not -1 < phi < 1:
        raise ValueError("phi deve essere in (-1, 1) perché la serie sia stazionaria.")

    noise = sigma * torch.randn(count - 1, generator=torch.Generator().manual_seed(seed),
                                dtype=torch.float64)
    returns = torch.empty_like(noise)
    previous = 0.0
    for index, shock in enumerate(noise.tolist()):
        previous = phi * previous + shock
        returns[index] = previous

    prices = 100.0 * torch.exp(torch.cat([torch.zeros(1, dtype=torch.float64), returns.cumsum(0)]))
    starts = [start + index * timedelta(days=1) for index in range(count)]
    return starts, prices
