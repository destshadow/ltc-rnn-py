import torch


def event_windows(events: dict[str, torch.Tensor], *, lookback: int) -> dict[str, torch.Tensor]:
    """Finestre degli ultimi `lookback` eventi e valori veri all'evento successivo.

    inputs   [n, lookback, 2]: rendimento e Δt di ogni evento (non standardizzati).
    targets  [n, 2]: drift e volatilità VERI in vigore all'evento successivo alla finestra.
    naive    [n, 2]: stime dalla sola finestra — drift = Σr / ΣΔt, volatilità = deviazione
             standard di (r − drift·Δt)/√Δt — il riferimento "da manuale".
    """
    if type(lookback) is not int or lookback < 2:
        raise ValueError("lookback deve essere un intero >= 2.")
    returns, dt = events["returns"].double(), events["dt"].double()
    count = len(returns) - lookback
    if count <= 0:
        raise ValueError("Troppi pochi eventi per la finestra richiesta.")

    window_returns = returns.unfold(0, lookback, 1)[:count]
    window_dt = dt.unfold(0, lookback, 1)[:count]

    naive_drift = window_returns.sum(dim=1) / window_dt.sum(dim=1)
    standardized = (window_returns - naive_drift[:, None] * window_dt) / window_dt.sqrt()
    naive_volatility = standardized.std(dim=1)

    return {
        "inputs": torch.stack([window_returns, window_dt], dim=2),
        "targets": torch.stack([events["true_drift"][lookback:],
                                events["true_volatility"][lookback:]], dim=1).double(),
        "naive": torch.stack([naive_drift, naive_volatility], dim=1),
    }
