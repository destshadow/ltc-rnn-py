"""Report sulla qualità dei dati prima del training: il training somiglia a ciò che si vuole prevedere?

    python -m experiments.data_quality_report --csv data/raw/gspc_daily_yahoo_20261119.csv \\
        --calendar weekdays --max-skipped 2 --by-decade

Di default mostra training e validazione; il test solo con --include-test, per non
usarlo nelle scelte.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from timeseries.calendar import CALENDARS, split_segments
from timeseries.data_quality import return_stats
from timeseries.dataset import returns_by_segment, split_pieces
from timeseries.prices import iter_csv_prices, resample_last


def print_row(name, stats, pieces=""):
    print(f"  {name:12s} {stats['count']:6d} {pieces:>6} {100 * stats['volatility']:7.2f}% "
          f"{stats['lag1_autocorrelation']:+7.3f} {100 * stats['max_abs']:7.2f}% "
          f"{100 * stats['extreme_share']:6.2f}%")


def decade_runs(pieces) -> dict[int, list[torch.Tensor]]:
    """Rendimenti per decennio, tenuti in tratti consecutivi (mai attraverso un buco)."""
    runs = defaultdict(list)
    for piece in pieces:
        decades = [time.year // 10 * 10 for time in piece.end_times]
        start = 0
        for index in range(1, len(decades) + 1):
            if index == len(decades) or decades[index] != decades[start]:
                runs[decades[start]].append(piece.returns[start:index])
                start = index
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--time-column", default="Date")
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--calendar", choices=CALENDARS, default="continuous")
    parser.add_argument("--max-skipped", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--by-decade", action="store_true", help="Anche per decennio (solo fuori dal test).")
    args = parser.parse_args()

    day = timedelta(days=1)
    starts, prices = resample_last(
        iter_csv_prices(args.csv, time_column=args.time_column, price_column=args.price_column),
        period=day,
    )
    segments = split_segments(starts, prices, period=day, calendar=args.calendar,
                              max_skipped=args.max_skipped)
    parts = dict(zip(("training", "validation", "test"), split_pieces(
        returns_by_segment(segments, period=day),
        validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
    )))
    if not args.include_test:
        test_start = parts.pop("test")[0].end_times[0]

    report = {"source": str(args.csv), "segments": len(segments), "parts": {}, "decades": {}}
    print(f"{args.csv.name}: {len(segments)} tratti senza dati mancanti ({args.calendar})")
    print(f"  {'parte':12s} {'giorni':>6} {'pezzi':>6} {'volat.':>8} {'autoc.':>7} "
          f"{'max |r|':>8} {'>4σ':>7}")
    for name, pieces in parts.items():
        stats = return_stats([piece.returns for piece in pieces])
        stats["first"], stats["last"] = f"{pieces[0].end_times[0]:%Y-%m-%d}", f"{pieces[-1].end_times[-1]:%Y-%m-%d}"
        report["parts"][name] = stats
        print_row(name, stats, str(len(pieces)))

    if args.by_decade:
        by_decade = decade_runs([piece for pieces in parts.values() for piece in pieces])
        print("  per decennio:")
        for decade in sorted(by_decade):
            stats = return_stats(by_decade[decade])
            report["decades"][str(decade)] = stats
            print_row(f"{decade}s", stats)

    if not args.include_test:
        print(f"  (test dal {test_start:%Y-%m-%d} escluso: usare --include-test solo per la verifica finale)")

    directory = Path("outputs") / f"data_quality_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report in {directory / 'report.json'}")


if __name__ == "__main__":
    main()
