import torch

from .dataset import Piece


def fit_ar1(pieces: list[Piece]) -> tuple[float, float]:
    """Stima r_{t+1} = a·r_t + b con i minimi quadrati sulle coppie consecutive.

    Le coppie si formano solo dentro ciascun pezzo, mai attraverso un'interruzione.
    È il riferimento lineare più semplice: se la LTC non lo batte, non migliora
    questa dipendenza dal rendimento precedente nella metrica e nel periodo valutati.
    """
    usable = [piece.returns for piece in pieces if len(piece.returns) > 1]
    # Controllo esplicito prima di concatenare: almeno due coppie consecutive.
    if sum(len(returns) - 1 for returns in usable) < 2:
        raise ValueError("Servono almeno due coppie di rendimenti consecutivi.")
    today = torch.cat([returns[:-1] for returns in usable])
    tomorrow = torch.cat([returns[1:] for returns in usable])

    today, tomorrow = today.double(), tomorrow.double()
    variance = (today - today.mean()).pow(2).sum()
    if not variance > 0:
        raise ValueError("Rendimenti costanti: coefficiente non stimabile.")

    slope = ((today - today.mean()) * (tomorrow - tomorrow.mean())).sum() / variance
    intercept = tomorrow.mean() - slope * today.mean()
    return slope.item(), intercept.item()


def window_last_returns(pieces: list[Piece], *, lookback: int) -> torch.Tensor:
    """Ultimo rendimento di ogni finestra (float64), nello stesso ordine di make_dataset."""
    values = [piece.returns[lookback - 1:-1] for piece in pieces if len(piece.returns) > lookback]
    if not values:
        raise ValueError("Nessun pezzo abbastanza lungo per la finestra richiesta.")
    return torch.cat(values).double()
