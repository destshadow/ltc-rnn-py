"""Grafico dei dati e della divisione training/validazione/test, prima del training.

    python -m experiments.plot_timeseries_split --csv data/raw/gspc_daily_yahoo_20261119.csv \\
        --calendar weekdays --max-skipped 2
"""

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib

from timeseries.calendar import CALENDARS, split_segments
from timeseries.dataset import returns_by_segment, split_pieces
from timeseries.prices import iter_csv_prices, resample_last


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--time-column", default="Date")
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--calendar", choices=CALENDARS, default="continuous")
    parser.add_argument("--max-skipped", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("outputs/plots/timeseries_split.png"))
    parser.add_argument("--show", action="store_true", help="Apre anche la finestra interattiva.")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.timeseries_split import plot_split

    period = timedelta(days=1)
    starts, prices = resample_last(
        iter_csv_prices(args.csv, time_column=args.time_column, price_column=args.price_column),
        period=period,
    )
    segments = split_segments(
        starts, prices, period=period, calendar=args.calendar, max_skipped=args.max_skipped,
    )
    training, validation, test = split_pieces(
        returns_by_segment(segments, period=period),
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )

    figure = plot_split(
        starts, prices, segments,
        {"training": training, "validation": validation, "test": test},
        title=f"{args.csv.name} — calendario {args.calendar}, max_skipped {args.max_skipped}, "
              f"{len(segments)} tratti",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=120)
    print(f"Grafico salvato in {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
