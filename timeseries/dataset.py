from dataclasses import dataclass
from datetime import datetime, timedelta

import torch

from .calendar import elapsed_periods
from .windows import Standardizer, chronological_split, log_returns, make_windows


@dataclass(frozen=True)
class Piece:
    """Rendimenti contigui: nessuna interruzione e un'unica parte della divisione.

    returns[j] ed elapsed[j] descrivono lo stesso intervallo, che termina in
    end_times[j].
    """

    returns: torch.Tensor
    elapsed: torch.Tensor
    end_times: list[datetime]


def returns_by_segment(
    segments: list[tuple[list[datetime], torch.Tensor]],
    *,
    period: timedelta,
) -> list[Piece]:
    """Rendimenti e durate calcolati dentro ciascun tratto, mai fra due tratti.

    I tratti con un solo prezzo non hanno rendimenti e vengono scartati.
    """
    pieces = []

    for starts, prices in segments:
        if len(starts) < 2:
            continue

        pieces.append(Piece(
            returns=log_returns(prices),
            elapsed=elapsed_periods(starts, period=period),
            end_times=list(starts[1:]),
        ))

    if not pieces:
        raise ValueError("Nessun tratto con almeno due prezzi.")

    return pieces


def _slice(piece: Piece, start: int, stop: int) -> Piece:
    return Piece(
        returns=piece.returns[start:stop],
        elapsed=piece.elapsed[start:stop],
        end_times=piece.end_times[start:stop],
    )


def split_pieces(
    pieces: list[Piece],
    *,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[Piece], list[Piece], list[Piece]]:
    """Divisione cronologica dei rendimenti di tutti i tratti.

    Le frazioni si applicano al numero totale di rendimenti; i confini sono
    istanti, quindi un tratto a cavallo di un confine viene tagliato in due.
    """
    end_times = [time for piece in pieces for time in piece.end_times]

    if any(later <= earlier for earlier, later in zip(end_times, end_times[1:])):
        raise ValueError("I tratti devono essere in ordine cronologico.")

    training, validation, _ = chronological_split(
        torch.arange(len(end_times)),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    validation_start = end_times[len(training)]
    test_start = end_times[len(training) + len(validation)]

    parts = ([], [], [])
    for piece in pieces:
        times = piece.end_times
        first_validation = sum(time < validation_start for time in times)
        first_test = sum(time < test_start for time in times)

        for part, (start, stop) in zip(parts, (
            (0, first_validation),
            (first_validation, first_test),
            (first_test, len(times)),
        )):
            if stop > start:
                part.append(_slice(piece, start, stop))

    return parts


def make_dataset(
    pieces: list[Piece],
    *,
    scaler: Standardizer,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finestre [n, lookback, 2] e obiettivi [n], create dentro ciascun pezzo.

    Ingresso 0: rendimento standardizzato. Ingresso 1: durata dello stesso
    intervallo, in periodi. Obiettivo: rendimento standardizzato successivo.
    I pezzi più corti di lookback + 1 non producono finestre.
    """
    all_inputs = []
    all_targets = []

    for piece in pieces:
        if len(piece.returns) <= lookback:
            continue

        standardized = scaler.transform(piece.returns)
        features = torch.stack([standardized, piece.elapsed], dim=1)
        inputs, targets = make_windows(
            features, lookback=lookback, targets=standardized
        )
        all_inputs.append(inputs)
        all_targets.append(targets)

    if not all_inputs:
        raise ValueError("Nessun pezzo abbastanza lungo per la finestra richiesta.")

    return (
        torch.cat(all_inputs).float(),
        torch.cat(all_targets).float(),
    )
