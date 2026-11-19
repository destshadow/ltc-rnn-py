"""Scarica lo storico giornaliero di un simbolo Yahoo Finance in data/raw/.

Richiede l'extra opzionale: python -m pip install -e '.[yahoo]'

    python -m experiments.download_yahoo --symbol ^GSPC

yfinance non è un'API ufficiale di Yahoo e le condizioni d'uso di Yahoo non
prevedono l'accesso automatico: uso personale e di studio, dati da non
ridistribuire. La scelta di usarlo è dell'utente.
"""

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def completed_rows(rows: list[tuple[date, float]], today: date) -> list[tuple[date, float]]:
    """Tiene solo le giornate chiuse: la riga di oggi (o successiva) può essere parziale."""
    rows = [(day, close) for day, close in rows if day < today]
    if not rows:
        raise ValueError("Nessuna giornata completa nello storico scaricato.")
    if any(later <= earlier for (earlier, _), (later, _) in zip(rows, rows[1:])):
        raise ValueError("Date non strettamente crescenti nello storico scaricato.")
    return rows


def to_csv(rows: list[tuple[date, float]]) -> bytes:
    lines = ["Date,Close"] + [f"{day.isoformat()},{close!r}" for day, close in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="^GSPC")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    import yfinance  # extra opzionale: importato solo qui

    history = yfinance.Ticker(args.symbol).history(
        period="max", interval="1d", auto_adjust=False,
    )
    rows = [(stamp.date(), float(close)) for stamp, close in history["Close"].dropna().items()]

    # Data di riferimento a New York: la giornata di borsa di oggi non è chiusa.
    today = datetime.now(ZoneInfo("America/New_York")).date()
    rows = completed_rows(rows, today)
    content = to_csv(rows)

    name = args.symbol.lstrip("^").lower()
    now = datetime.now(timezone.utc)
    path = args.output_dir / f"{name}_daily_yahoo_{now:%Y%m%d}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:  # ogni versione resta, nessuna sovrascrittura
        file.write(content)

    provenance = {
        "source": "Yahoo Finance via yfinance (non ufficiale)",
        "symbol": args.symbol,
        "yfinance_version": yfinance.__version__,
        "downloaded_at": now.isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(rows),
        "first_date": rows[0][0].isoformat(),
        "last_date": rows[-1][0].isoformat(),
    }
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"{path}: {len(rows)} giornate, {rows[0][0]} → {rows[-1][0]}")


if __name__ == "__main__":
    main()
