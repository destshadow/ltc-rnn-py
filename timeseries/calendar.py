from datetime import datetime, timedelta

import torch


# continuous: ogni periodo è atteso (cripto, 24/7).
# weekdays: sabato e domenica chiusi (azioni); festività non incluse.
CALENDARS = ("continuous", "weekdays")


def _validate_calendar(period: timedelta, calendar: str) -> None:
    if period <= timedelta(0):
        raise ValueError("period deve essere positivo.")

    if calendar not in CALENDARS:
        raise ValueError(f"Calendario non supportato: {calendar!r}.")

    if calendar == "weekdays" and period != timedelta(days=1):
        raise ValueError("Il calendario 'weekdays' richiede period di un giorno.")


def expected_periods(
    start: datetime,
    end: datetime,
    *,
    period: timedelta,
    calendar: str,
) -> int:
    """Periodi di apertura attesi in (start, end]; 1 se end è il successivo."""
    _validate_calendar(period, calendar)

    if end <= start:
        raise ValueError("end deve essere successivo a start.")

    if calendar == "continuous":
        return (end - start) // period

    weeks, rest = divmod((end.date() - start.date()).days, 7)
    weekday = start.weekday()

    return weeks * 5 + sum((weekday + k) % 7 < 5 for k in range(1, rest + 1))


def split_segments(
    starts: list[datetime],
    prices: torch.Tensor,
    *,
    period: timedelta,
    calendar: str,
    max_skipped: int = 0,
) -> list[tuple[list[datetime], torch.Tensor]]:
    """Spezza la serie dove mancano più di max_skipped periodi attesi.

    Le chiusure previste dal calendario non spezzano la serie; le altre pause
    sono dati mancanti e nessuna finestra deve attraversarle. Nessun prezzo
    viene inserito. Restituisce i tratti in ordine, anche se brevi.
    """
    _validate_calendar(period, calendar)

    if type(max_skipped) is not int or max_skipped < 0:
        raise ValueError("max_skipped deve essere un intero non negativo.")

    if len(starts) == 0 or len(starts) != len(prices):
        raise ValueError("Servono istanti e prezzi non vuoti, uno per prezzo.")

    if calendar == "weekdays":
        for start in starts:
            if start.weekday() >= 5:
                raise ValueError(f"Osservazione in un giorno di chiusura: {start}.")

    segments = []
    begin = 0

    for index in range(1, len(starts)):
        skipped = expected_periods(
            starts[index - 1], starts[index], period=period, calendar=calendar
        ) - 1

        if skipped > max_skipped:
            segments.append((starts[begin:index], prices[begin:index]))
            begin = index

    segments.append((starts[begin:], prices[begin:]))

    return segments


def elapsed_periods(
    starts: list[datetime],
    *,
    period: timedelta,
) -> torch.Tensor:
    """Durata di calendario fra osservazioni consecutive, in periodi: [n - 1].

    Da affiancare ai rendimenti come ingresso: venerdì -> lunedì vale 3.0.
    """
    if period <= timedelta(0):
        raise ValueError("period deve essere positivo.")

    if len(starts) < 2:
        raise ValueError("Servono almeno due istanti.")

    durations = [
        (current - previous) / period
        for previous, current in zip(starts, starts[1:])
    ]

    if min(durations) <= 0:
        raise ValueError("Gli istanti devono essere strettamente crescenti.")

    return torch.tensor(durations, dtype=torch.float64)
