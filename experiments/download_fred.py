"""Scarica una serie giornaliera da FRED in data/raw/, con impronta e provenienza.

    python -m experiments.download_fred --series SP500
"""

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def check_fred_csv(content: bytes, series: str) -> dict:
    """Verifica il formato di FRED e conta righe e valori presenti."""
    lines = content.decode("utf-8").strip().splitlines()

    if not lines or lines[0].strip() != f"observation_date,{series}":
        raise ValueError("Risposta inattesa da FRED: intestazione non valida.")

    rows = [line.split(",") for line in lines[1:]]
    if any(len(row) != 2 for row in rows):
        raise ValueError("Risposta inattesa da FRED: righe con un numero di campi errato.")

    # FRED lascia vuoti (o ".") i giorni senza valore, per esempio le festività.
    present = sum(row[1].strip() not in ("", ".") for row in rows)
    if present == 0:
        raise ValueError("La serie non contiene valori.")

    return {"rows": len(rows), "values": present}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", default="SP500")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    url = URL.format(series=args.series)
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()

    counts = check_fred_csv(content, args.series)
    now = datetime.now(timezone.utc)
    path = args.output_dir / f"{args.series.lower()}_daily_fred_{now:%Y%m%d}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ogni versione resta riproducibile: non si sovrascrive un file esistente.
    with path.open("xb") as file:
        file.write(content)

    provenance = {
        "url": url,
        "downloaded_at": now.isoformat(),
        "sha256": hashlib.sha256(content).hexdigest(),
        **counts,
    }
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"{path}: {counts['values']} valori su {counts['rows']} righe")


if __name__ == "__main__":
    main()
