import csv
import math
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

import torch


def _parse_price(text: str) -> float | None:
    """Restituisce il prezzo, oppure None se mancante o non valido."""
    text = text.strip()
    if text in ("", "?"):
        return None

    try:
        value = float(text)
    except ValueError:
        return None

    if not math.isfinite(value) or value <= 0:
        return None

    return value


def iter_csv_prices(
    path,
    *,
    time_column: str = "Date",
    price_column: str = "Close",
    time_format: str | None = None,
) -> Iterator[tuple[datetime, float]]:
    """Legge (istante, prezzo) da un CSV con intestazione, saltando i mancanti."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for name in (time_column, price_column):
            if reader.fieldnames is None or name not in reader.fieldnames:
                raise ValueError(f"Colonna assente nel CSV: {name}.")

        for row in reader:
            price = _parse_price(row[price_column])
            if price is None:
                continue

            text = row[time_column].strip()
            time = (
                datetime.strptime(text, time_format)
                if time_format
                else datetime.fromisoformat(text)
            )
            yield time, price


def iter_arff_prices(
    path,
    *,
    time_attribute: str = "Timestamp",
    price_attribute: str = "Close",
) -> Iterator[tuple[datetime, float]]:
    """Legge (istante, prezzo) da un file ARFF; '?' indica un valore mancante."""
    attributes = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            keyword = stripped.split(maxsplit=1)[0].upper()
            if keyword == "@ATTRIBUTE":
                attributes.append(stripped.split()[1])
            elif keyword == "@DATA":
                break
        else:
            raise ValueError("Sezione @DATA assente nel file ARFF.")

        for name in (time_attribute, price_attribute):
            if name not in attributes:
                raise ValueError(f"Attributo assente nel file ARFF: {name}.")

        time_index = attributes.index(time_attribute)
        price_index = attributes.index(price_attribute)

        for row in csv.reader(file, quotechar="'"):
            if not row or row[0].startswith("%"):
                continue

            price = _parse_price(row[price_index])
            if price is None:
                continue

            yield datetime.fromisoformat(row[time_index].strip()), price


def resample_last(
    samples: Iterable[tuple[datetime, float]],
    *,
    period: timedelta,
) -> tuple[list[datetime], torch.Tensor]:
    """Tiene l'ultimo prezzo di ogni periodo (es. chiusura giornaliera).

    I periodi senza dati non compaiono e nessun prezzo viene inserito: le pause
    restano visibili negli inizi restituiti. Chiusure di mercato e dati mancanti
    si distinguono con timeseries.calendar (split_segments, elapsed_periods).
    Restituisce l'inizio di ogni periodo e i prezzi float64. Gli istanti devono
    essere strettamente crescenti.
    """
    if period <= timedelta(0):
        raise ValueError("period deve essere positivo.")

    origin = datetime(1970, 1, 1)
    starts = []
    prices = []
    previous = None

    for time, price in samples:
        if time.tzinfo is not None:
            raise ValueError("Usare istanti senza fuso orario, nello stesso riferimento.")

        if previous is not None and time <= previous:
            raise ValueError("Gli istanti devono essere strettamente crescenti.")
        previous = time

        start = origin + ((time - origin) // period) * period

        if starts and starts[-1] == start:
            prices[-1] = price
        else:
            starts.append(start)
            prices.append(price)

    if not prices:
        raise ValueError("Nessun prezzo valido trovato.")

    return starts, torch.tensor(prices, dtype=torch.float64)
