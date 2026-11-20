import torch

from .dataset import Piece
from .windows import Standardizer


def volatility_dataset(
    pieces: list[Piece],
    *,
    scaler: Standardizer,
    lookback: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finestre per prevedere la volatilità realizzata dei prossimi `horizon` giorni.

    Ingressi [n, lookback, 2]: rendimento standardizzato e durata, come make_dataset.
    Obiettivo [n] (float64): media di |rendimento| sui `horizon` rendimenti successivi
    alla finestra, nello stesso pezzo (mai oltre un buco o un confine della divisione).
    Persistenza [n] (float64): media di |rendimento| degli ultimi `horizon` della
    finestra, il riferimento "la volatilità resta quella recente".
    """
    for name, value in (("lookback", lookback), ("horizon", horizon)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} deve essere un intero positivo.")
    if horizon > lookback:
        raise ValueError("horizon non può superare lookback (serve per la persistenza).")

    inputs, targets, persistence = [], [], []
    for piece in pieces:
        count = len(piece.returns) - lookback - horizon + 1
        if count <= 0:
            continue

        absolute = piece.returns.double().abs()
        features = torch.stack([scaler.transform(piece.returns), piece.elapsed], dim=1)
        # Medie mobili di |r| su `horizon` giorni: rolling[k] = media di absolute[k:k+horizon].
        rolling = absolute.unfold(0, horizon, 1).mean(dim=1)

        inputs.append(features.unfold(0, lookback, 1)[:count].transpose(1, 2))
        targets.append(rolling[lookback:lookback + count])
        persistence.append(rolling[lookback - horizon:lookback - horizon + count])

    if not inputs:
        raise ValueError("Nessun pezzo abbastanza lungo per lookback + horizon.")

    return torch.cat(inputs).float(), torch.cat(targets), torch.cat(persistence)


# Pavimento per evitare log(0) se una finestra ha solo rendimenti nulli.
VOLATILITY_FLOOR = 1e-8


def to_log_ratio(targets: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
    """Obiettivo come correzione della persistenza: log(volatilità futura / persistenza)."""
    if targets.shape != persistence.shape:
        raise ValueError("targets e persistence devono avere la stessa forma.")
    return torch.log(targets.double().clamp_min(VOLATILITY_FLOOR)) - torch.log(
        persistence.double().clamp_min(VOLATILITY_FLOOR))


def from_log_ratio(log_ratio: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
    """Previsione = persistenza × e^(correzione): i picchi li segue la persistenza."""
    if log_ratio.shape != persistence.shape:
        raise ValueError("log_ratio e persistence devono avere la stessa forma.")
    return persistence.double().clamp_min(VOLATILITY_FLOOR) * torch.exp(log_ratio.double())
