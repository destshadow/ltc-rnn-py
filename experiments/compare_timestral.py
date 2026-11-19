"""Timestral sullo stesso test della LTC: previsione centrale e quantili a un passo.

Richiede l'extra opzionale: python -m pip install -e '.[timestral]'

    python -m experiments.compare_timestral --csv data/raw/gspc_daily_yahoo_20261119.csv \\
        --calendar weekdays --max-skipped 2
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from timeseries.calendar import CALENDARS, split_segments
from timeseries.dataset import make_dataset, returns_by_segment, split_pieces
from timeseries.evaluation import baseline_predictions, regression_report
from timeseries.prices import iter_csv_prices, resample_last
from timeseries.quantile_metrics import constant_quantiles, interval_coverage, pinball_loss
from timeseries.timestral_baseline import quantile_forecasts
from timeseries.windows import Standardizer

# Revisione dei pesi su Hugging Face, fissata per la riproducibilità.
REVISIONS = {"mistralai/Timestral-5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"}


def quantile_report(quantiles, levels, targets):
    return {
        "pinball": pinball_loss(quantiles, levels, targets),
        "band_10_90": interval_coverage(quantiles, levels, targets, lower=0.1, upper=0.9),
        "band_05_95": interval_coverage(quantiles, levels, targets, lower=0.05, upper=0.95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--time-column", default="Date")
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--calendar", choices=CALENDARS, default="continuous")
    parser.add_argument("--max-skipped", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=30, help="Stesse finestre della LTC.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--model", default="mistralai/Timestral-5M", choices=sorted(REVISIONS))
    args = parser.parse_args()

    from timestral import TimestralForecaster  # extra opzionale

    # Stessa preparazione di experiments.train_timeseries: stessi obiettivi di test.
    day = timedelta(days=1)
    starts, prices = resample_last(
        iter_csv_prices(args.csv, time_column=args.time_column, price_column=args.price_column),
        period=day,
    )
    segments = split_segments(starts, prices, period=day, calendar=args.calendar,
                              max_skipped=args.max_skipped)
    training, _, test = split_pieces(returns_by_segment(segments, period=day),
                                     validation_fraction=args.validation_fraction,
                                     test_fraction=args.test_fraction)
    training_returns = torch.cat([piece.returns for piece in training])
    scaler = Standardizer.fit(training_returns)
    inputs, targets = make_dataset(test, scaler=scaler, lookback=args.lookback)
    windows = scaler.inverse(inputs[:, :, 0].double()).float()   # rendimenti originali
    targets = scaler.inverse(targets.double())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forecaster = TimestralForecaster.from_pretrained(
        args.model, device=device, revision=REVISIONS[args.model],
    )
    quantiles, levels = quantile_forecasts(forecaster, windows)
    median = quantiles[:, levels.index(0.5)].double()

    point = {"timestral_median": regression_report(median, targets)}
    for name, values in baseline_predictions(targets, training_mean=scaler.mean).items():
        point[name] = regression_report(values, targets)
    historical = constant_quantiles(training_returns, levels, len(targets))
    quantile = {
        "timestral": quantile_report(quantiles.double(), levels, targets),
        "training_distribution": quantile_report(historical, levels, targets),
    }

    print(f"Test: {len(targets)} finestre, {args.model}")
    for name, values in point.items():
        print(f"  {name:16s} mse {values['mse']:.3e} | mae {values['mae']:.3e}")
    for name, values in quantile.items():
        band = values["band_10_90"]
        print(f"  {name:22s} pinball {values['pinball']:.3e} | fascia 10–90: copertura "
              f"{100 * band['coverage']:.1f}%, larghezza {band['mean_width']:.4f}")

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path("outputs") / f"compare_timestral_{run}"
    directory.mkdir(parents=True, exist_ok=False)
    protocol = {"source": str(args.csv), "model": args.model, "revision": REVISIONS[args.model],
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "test_windows": len(targets), "torch_version": torch.__version__, "device": device}
    torch.save({"quantiles": quantiles, "levels": levels, "targets": targets}, directory / "forecasts.pt")
    (directory / "results.json").write_text(
        json.dumps({"protocol": protocol, "point": point, "quantile": quantile},
                   indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Risultati in {directory}")


if __name__ == "__main__":
    main()
