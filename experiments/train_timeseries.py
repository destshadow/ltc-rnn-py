"""Previsione del rendimento successivo con la LTC, confrontata con i riferimenti.

Esempio (S&P 500 giornaliero; argomenti in experiments/timeseries_args.py):

    python -m experiments.train_timeseries --csv data/raw/gspc_daily_yahoo_20261119.csv \\
        --calendar weekdays --max-skipped 2 --epochs 20
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from experiments.timeseries_args import parse_arguments
from experiments.timeseries_data import load_parts
from timeseries.dataset import INPUTS, make_dataset, window_targets
from timeseries.evaluation import baseline_predictions, predict, regression_report
from timeseries.linear_baseline import fit_ar1, window_last_returns
from timeseries.model import make_forecaster
from timeseries.significance import diebold_mariano_or_unavailable
from training.checkpoint import save_atomically
from training.fit import fit_and_select
from training.metrics_log import MetricsLog



def report_all(name, model, inputs, targets, scaler, args, training_mean, linear):
    """Confronta LTC e riferimenti nelle unità originali dei rendimenti.

    targets: rendimenti originali (window_targets), non ricostruiti dallo scaler.
    Restituisce le metriche (con Diebold–Mariano LTC contro ogni riferimento)
    e le previsioni, da salvare per analisi successive.
    """
    slope, intercept, last_returns = linear  # riferimento lineare r_{t+1} = a·r_t + b
    forecasts = {
        "ltc": scaler.inverse(predict(model, inputs, dt=args.dt).double()),
        **baseline_predictions(targets, training_mean=training_mean),
        "linear_ar1": slope * last_returns + intercept,
    }
    results = {label: regression_report(values, targets) for label, values in forecasts.items()}
    errors = {label: (values - targets) ** 2 for label, values in forecasts.items()}
    significance = {label: diebold_mariano_or_unavailable(errors["ltc"], errors[label])
                    for label in forecasts if label != "ltc"}

    print(f"\n{name} ({len(targets)} finestre)")
    for label, values in results.items():
        direction = values["direction_accuracy"]
        p_value = None if label == "ltc" else significance[label]["p_value"]
        versus = "" if label == "ltc" else (
            " | LTC vs: p=n/d" if p_value is None else f" | LTC vs: p={p_value:.3f}")
        print(
            f"  {label:14s} mse {values['mse']:.3e} | "
            f"mae {values['mae']:.3e} | segno "
            + ("n/d" if direction is None else f"{100 * direction:.1f}%") + versus
        )

    spread = results["ltc"]["spread_ratio"]
    if spread is not None and spread < 0.05:  # soglia di avviso, non un criterio
        print(f"  attenzione: previsioni LTC quasi costanti (oscillazione {spread:.1%} degli obiettivi)")
    return {**results, "ltc_significance": significance}, forecasts


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Lettura e ricampionamento dei prezzi...", flush=True)
    starts, prices, segments, parts, scaler = load_parts(args)

    datasets = {
        name: make_dataset(pieces, scaler=scaler, lookback=args.lookback)
        for name, pieces in zip(("training", "validation", "test"), parts)
    }

    # Ingressi: rendimento standardizzato e durata dell'intervallo in periodi.
    config, model = make_forecaster(
        args.cell, input_size=len(INPUTS), hidden_size=args.hidden_size, substeps=args.substeps,
        backbone_units=args.backbone_units, elapsed_channel=INPUTS.index("elapsed_periods"),
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    training_inputs, training_targets = (
        tensor.to(device) for tensor in datasets["training"]
    )
    validation_inputs, validation_targets = datasets["validation"]

    print("Dispositivo:", device)
    print(f"Periodi: {len(prices)} ({starts[0]} → {starts[-1]})")
    print(f"Tratti senza dati mancanti ({args.calendar}): {len(segments)}")
    print("Finestre:", {name: len(data[1]) for name, data in datasets.items()})

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path("outputs") / f"timeseries_{run_name}"
    directory.mkdir(parents=True, exist_ok=False)
    # Riferimento: prevedere sempre la media del training (0 standardizzato).
    metrics = MetricsLog(
        directory / "metrics.csv",
        validation_baseline=validation_targets.pow(2).mean().item(),
    )
    print("Metriche dal vivo:", directory / "metrics.csv", flush=True)

    # Il miglior modello va su disco appena migliora: un'interruzione non lo perde.
    checkpoint = {
        "config": asdict(config), "cell": args.cell, "dt": args.dt, "lookback": args.lookback,
        "standardizer": asdict(scaler), "calendar": args.calendar,
        "period_hours": args.period_hours, "max_skipped": args.max_skipped,
        "inputs": INPUTS,
    }
    best_epoch, best_state = fit_and_select(
        model, optimizer, loss_function,
        (training_inputs, training_targets), (validation_inputs, validation_targets),
        dt=args.dt, epochs=args.epochs, batch_size=args.batch_size,
        final_epoch=args.final_epoch, metrics=metrics, predict=predict,
        save_best=lambda state, epoch, loss: save_atomically(
            {**checkpoint, "model_state_dict": state, "epoch": epoch, "validation_loss": loss},
            directory / "best.pt"),
    )

    # Il test viene valutato una sola volta, con il modello scelto sulla validazione.
    model.load_state_dict(best_state)
    print(f"\nEpoca usata: {best_epoch}"
          + (" (ultima, --final-epoch)" if args.final_epoch else " (migliore in validazione)"))

    ar1 = fit_ar1(parts[0])  # stimato solo sul training
    print(f"Riferimento lineare: r(t+1) = {ar1[0]:+.4f}·r(t) {ar1[1]:+.2e}")
    reports = {
        name: report_all(
            name, model, datasets[name][0],
            window_targets(parts[index], lookback=args.lookback),
            scaler, args, scaler.mean,
            (*ar1, window_last_returns(parts[index], lookback=args.lookback)),
        )
        for index, name in ((1, "validation"), (2, "test"))
    }
    results = {name: report[0] for name, report in reports.items()}
    torch.save({name: {**report[1], "targets": window_targets(parts[index], lookback=args.lookback)}
                for index, (name, report) in enumerate(reports.items(), start=1)},
               directory / "predictions.pt")

    protocol = {
        "source": str(args.csv or args.arff),
        "arguments": {
            key: value if value is None or isinstance(value, (bool, int, float, str)) else str(value)
            for key, value in vars(args).items()
        },
        "periods": len(prices),
        "first_period": starts[0].isoformat(),
        "last_period": starts[-1].isoformat(),
        "segments": len(segments),
        "inputs": INPUTS,
        "standardizer": asdict(scaler),
        "best_epoch": best_epoch,
        "torch_version": torch.__version__,
        "device": str(device),
    }

    (directory / "results.json").write_text(
        json.dumps(
            {"protocol": protocol, "results": results}, indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"\nRisultati in {directory}")


if __name__ == "__main__":
    main()
