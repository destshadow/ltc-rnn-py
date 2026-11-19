import torch


def quantile_forecasts(
    forecaster,
    windows: torch.Tensor,
    *,
    batch_size: int = 256,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Quantili a un passo di Timestral per ogni finestra: [n, lookback] -> [n, Q].

    forecaster è un TimestralForecaster già caricato. Restituisce i quantili
    sulla CPU, nello stesso ordine delle finestre, e i loro livelli.
    """
    if windows.ndim != 2 or 0 in windows.shape:
        raise ValueError("windows deve avere forma [n, lookback] con n e lookback positivi.")

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size deve essere un intero positivo.")

    outputs = []
    levels = None

    for start in range(0, len(windows), batch_size):
        forecast = forecaster.forecast(
            windows[start:start + batch_size],
            prediction_length=1,
            # I rendimenti possono essere negativi anche dopo un contesto di soli rialzi.
            clip_nonnegative=False,
        )
        outputs.append(forecast.quantiles[:, 0, 0, :].cpu())
        levels = forecast.quantile_levels

    return torch.cat(outputs), levels
