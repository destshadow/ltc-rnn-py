import math

import torch


def diebold_mariano(loss_a: torch.Tensor, loss_b: torch.Tensor, *, lags: int = 5) -> dict:
    """Test di Diebold–Mariano: la differenza di errore medio fra A e B è casuale?

    loss_a, loss_b: errore per esempio (es. quadratico) sugli stessi obiettivi, in
    ordine temporale. d = loss_a − loss_b; mean_difference < 0 significa A migliore.
    La varianza della media usa Newey–West con `lags` ritardi, perché gli errori
    dei rendimenti sono correlati nel tempo (periodi agitati e calmi).
    p_value: bilaterale, approssimazione normale. Non corregge per scelte fatte
    guardando gli stessi dati (es. epoca scelta su questa validazione).
    """
    if loss_a.shape != loss_b.shape or loss_a.ndim != 1:
        raise ValueError("Servono due serie di errori [n] della stessa forma.")
    if type(lags) is not int or lags < 0:
        raise ValueError("lags deve essere un intero non negativo.")
    if len(loss_a) <= lags + 1:
        raise ValueError("Troppi pochi esempi per il numero di ritardi.")

    difference = loss_a.double() - loss_b.double()
    count = len(difference)
    centered = difference - difference.mean()

    variance = centered.pow(2).mean()
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)  # Bartlett
        variance = variance + 2 * weight * (centered[lag:] * centered[:-lag]).mean()

    if not variance > 0:
        raise ValueError("Differenza di errore costante: test non definito.")

    statistic = (difference.mean() / torch.sqrt(variance / count)).item()
    p_value = math.erfc(abs(statistic) / math.sqrt(2))
    return {
        "count": count,
        "mean_difference": difference.mean().item(),
        "statistic": statistic,
        "p_value": p_value,
    }
