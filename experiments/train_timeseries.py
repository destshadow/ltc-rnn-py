"""Previsione del rendimento successivo con la LTC, confrontata con i riferimenti.

Esempio (dataset Bitstamp al minuto, chiusure giornaliere):

    python -m experiments.train_timeseries --arff "../BTC dataset/dataset_"
"""

import argparse
import copy
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from torch import nn

from ltc.config import LTCConfig
from timeseries.evaluation import baseline_predictions, predict, regression_report
from timeseries.model import ReturnForecaster
from timeseries.prices import iter_arff_prices, iter_csv_prices, resample_last
from timeseries.windows import (
    Standardizer,
    chronological_split,
    log_returns,
    make_windows,
)
from training.epoch import train_epoch


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV con intestazione.")
    source.add_argument("--arff", type=Path, help="File ARFF, es. Bitstamp.")
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--period-hours", type=float, default=24.0)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_series(args):
    if args.csv is not None:
        samples = iter_csv_prices(
            args.csv,
            time_column=args.time_column or "Date",
            price_column=args.price_column,
        )
    else:
        samples = iter_arff_prices(
            args.arff,
            time_attribute=args.time_column or "Timestamp",
            price_attribute=args.price_column,
        )

    return resample_last(samples, period=timedelta(hours=args.period_hours))


def report_all(name, model, inputs, targets, scaler, args, training_mean):
    """Confronta LTC e riferimenti nelle unità originali dei rendimenti."""
    predictions = scaler.inverse(predict(model, inputs, dt=args.dt))
    targets = scaler.inverse(targets)

    results = {"ltc": regression_report(predictions, targets)}
    for baseline, values in baseline_predictions(
        targets, training_mean=training_mean
    ).items():
        results[baseline] = regression_report(values, targets)

    print(f"\n{name} ({len(targets)} finestre)")
    for label, values in results.items():
        direction = values["direction_accuracy"]
        print(
            f"  {label:14s} mse {values['mse']:.3e} | "
            f"mae {values['mae']:.3e} | segno "
            + ("n/d" if math.isnan(direction) else f"{100 * direction:.1f}%")
        )

    return results


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Lettura e ricampionamento dei prezzi...", flush=True)
    starts, prices = load_series(args)
    returns = log_returns(prices)

    training, validation, test = chronological_split(
        returns,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    scaler = Standardizer.fit(training)

    datasets = {}
    for name, segment in (
        ("training", training), ("validation", validation), ("test", test),
    ):
        inputs, targets = make_windows(
            scaler.transform(segment).float(), lookback=args.lookback
        )
        datasets[name] = (inputs, targets)

    config = LTCConfig(
        input_size=1, hidden_size=args.hidden_size, substeps=args.substeps
    )
    model = ReturnForecaster(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    training_inputs, training_targets = (
        tensor.to(device) for tensor in datasets["training"]
    )
    validation_inputs, validation_targets = datasets["validation"]

    print("Dispositivo:", device)
    print(f"Periodi: {len(prices)} ({starts[0]} → {starts[-1]})")
    print("Finestre:", {name: len(data[1]) for name, data in datasets.items()})

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            train_epoch(
                model, optimizer, loss_function,
                training_inputs, training_targets,
                dt=args.dt, batch_size=args.batch_size,
            )

        if epoch % 10 != 0 and epoch != args.epochs:
            continue

        predictions = predict(model, validation_inputs, dt=args.dt)
        validation_loss = loss_function(predictions, validation_targets).item()
        print(f"Epoca {epoch:4d} | val mse (standardizzato) {validation_loss:.4f}")

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Il test viene valutato una sola volta, con il modello scelto sulla validazione.
    model.load_state_dict(best_state)
    print(f"\nMigliore epoca sulla validazione: {best_epoch}")

    results = {
        name: report_all(
            name, model, *datasets[name], scaler, args, scaler.mean
        )
        for name in ("validation", "test")
    }

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path("outputs") / f"timeseries_{run_name}"
    directory.mkdir(parents=True, exist_ok=False)

    protocol = {
        "source": str(args.csv or args.arff),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "periods": len(prices),
        "first_period": starts[0].isoformat(),
        "last_period": starts[-1].isoformat(),
        "standardizer": asdict(scaler),
        "best_epoch": best_epoch,
        "torch_version": torch.__version__,
        "device": str(device),
    }

    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "dt": args.dt,
            "lookback": args.lookback,
            "standardizer": asdict(scaler),
        },
        directory / "best.pt",
    )
    (directory / "results.json").write_text(
        json.dumps({"protocol": protocol, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nRisultati in {directory}")


if __name__ == "__main__":
    main()
