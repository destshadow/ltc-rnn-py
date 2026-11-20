"""Previsione della volatilità realizzata dei prossimi giorni con la LTC.

Obiettivo: media di |rendimento| sui prossimi --horizon giorni. Riferimenti:
persistenza (volatilità degli ultimi giorni) e volatilità media del training.

    python -m experiments.train_volatility --csv data/raw/gspc_daily_yahoo_20261119.csv \\
        --calendar weekdays --max-skipped 2 --train-start 1990-01-01 \\
        --validation-start 2004-01-01 --test-start 2012-01-27 --epochs 30 --final-epoch
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from experiments.timeseries_args import build_parser, parse_arguments
from experiments.timeseries_data import load_parts
from ltc.config import LTCConfig
from timeseries.dataset import INPUTS
from timeseries.evaluation import predict, regression_report, tail_split_mse
from timeseries.model import ReturnForecaster
from timeseries.significance import diebold_mariano_or_unavailable
from timeseries.volatility import from_log_ratio, to_log_ratio, volatility_dataset
from timeseries.windows import Standardizer
from training.checkpoint import save_atomically
from training.fit import fit_and_select
from training.metrics_log import MetricsLog


def report(name, model, part, target_scaler, args, training_volatility):
    """LTC contro persistenza e volatilità media, nelle unità originali (|rendimento|)."""
    inputs, targets, persistence = part
    output = target_scaler.inverse(predict(model, inputs, dt=args.dt).double())
    forecasts = {
        # persistence_ratio: la rete prevede log(futura/persistenza), quindi corregge la persistenza.
        "ltc": from_log_ratio(output, persistence) if args.mode == "persistence_ratio" else output,
        "persistence": persistence,
        "training_mean": torch.full_like(targets, training_volatility),
    }
    results = {label: {**regression_report(values, targets), "tail": tail_split_mse(values, targets)}
               for label, values in forecasts.items()}
    errors = {label: (values - targets) ** 2 for label, values in forecasts.items()}
    # Obiettivi su giorni che si sovrappongono: errori correlati, più ritardi nel test.
    lags = max(5, 2 * args.horizon)
    results["ltc_significance"] = {
        label: diebold_mariano_or_unavailable(errors["ltc"], errors[label], lags=lags)
        for label in forecasts if label != "ltc"
    }

    print(f"\n{name} ({len(targets)} finestre, orizzonte {args.horizon} giorni)")
    for label in forecasts:
        values = results[label]
        test = results["ltc_significance"].get(label, {})
        p_text = "" if label == "ltc" else (
            " | LTC vs: p=n/d" if test.get("p_value") is None else f" | LTC vs: p={test['p_value']:.3f}")
        correlation = values["correlation"]
        print(f"  {label:14s} mse {values['mse']:.3e} | mae {values['mae']:.3e} | corr "
              + ("n/d" if correlation is None else f"{correlation:+.3f}") + p_text)
    return results, {**forecasts, "targets": targets}


def main():
    parser = build_parser("Previsione della volatilità realizzata con la LTC.")
    parser.add_argument("--horizon", type=int, default=5, help="Giorni futuri della volatilità.")
    parser.add_argument("--mode", choices=("level", "persistence_ratio"), default="level",
                        help="level: volatilità diretta; persistence_ratio: correzione della persistenza.")
    args = parse_arguments(parser=parser)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Lettura e ricampionamento dei prezzi...", flush=True)
    _, _, segments, parts, scaler = load_parts(args)
    datasets = {name: volatility_dataset(pieces, scaler=scaler, lookback=args.lookback,
                                         horizon=args.horizon)
                for name, pieces in zip(("training", "validation", "test"), parts)}
    # La rete impara la volatilità standardizzata sul training; i report tornano alle unità originali.
    def raw_target(name):
        _, targets, persistence = datasets[name]
        return to_log_ratio(targets, persistence) if args.mode == "persistence_ratio" else targets

    target_scaler = Standardizer.fit(raw_target("training"))
    training_volatility = datasets["training"][1].mean().item()

    def standardized(name):
        return datasets[name][0], target_scaler.transform(raw_target(name)).float()

    config = LTCConfig(input_size=2, hidden_size=args.hidden_size, substeps=args.substeps)
    model = ReturnForecaster(config).to(device)
    training = tuple(tensor.to(device) for tensor in standardized("training"))
    validation = standardized("validation")
    print("Dispositivo:", device, "| tratti:", len(segments),
          "| finestre:", {name: len(data[1]) for name, data in datasets.items()})

    directory = Path("outputs") / f"volatility_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}"
    directory.mkdir(parents=True, exist_ok=False)
    metrics = MetricsLog(directory / "metrics.csv",
                         validation_baseline=validation[1].double().pow(2).mean().item())
    print("Metriche dal vivo:", directory / "metrics.csv", flush=True)
    checkpoint = {"config": asdict(config), "dt": args.dt, "lookback": args.lookback,
                  "horizon": args.horizon, "mode": args.mode,
                  "target": "realized_volatility_mean_abs_return",
                  "standardizer": asdict(scaler), "target_standardizer": asdict(target_scaler),
                  # Contratto temporale: serve a ricostruire finestre e durate dai soli prezzi.
                  "period_hours": args.period_hours, "calendar": args.calendar,
                  "max_skipped": args.max_skipped, "inputs": INPUTS}

    best_epoch, best_state = fit_and_select(
        model, torch.optim.Adam(model.parameters(), lr=args.learning_rate), nn.MSELoss(),
        training, validation, dt=args.dt, epochs=args.epochs, batch_size=args.batch_size,
        final_epoch=args.final_epoch, metrics=metrics, predict=predict,
        save_best=lambda state, epoch, loss: save_atomically(
            {**checkpoint, "model_state_dict": state, "epoch": epoch, "validation_loss": loss},
            directory / "best.pt"),
    )
    model.load_state_dict(best_state)
    print(f"\nEpoca usata: {best_epoch}" + (" (ultima)" if args.final_epoch else " (migliore in validazione)"))

    reports = {name: report(name, model, datasets[name], target_scaler, args, training_volatility)
               for name in ("validation", "test")}
    torch.save({name: forecasts for name, (_, forecasts) in reports.items()}, directory / "predictions.pt")
    protocol = {"source": str(args.csv or args.arff), "best_epoch": best_epoch,
                "arguments": {key: value if value is None or isinstance(value, (bool, int, float, str))
                              else str(value) for key, value in vars(args).items()},
                "segments": len(segments), "training_volatility": training_volatility,
                "torch_version": torch.__version__, "device": str(device)}
    (directory / "results.json").write_text(json.dumps(
        {"protocol": protocol, "results": {name: results for name, (results, _) in reports.items()}},
        indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nRisultati in {directory}")


if __name__ == "__main__":
    main()
