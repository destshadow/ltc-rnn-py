"""Controllo del modello per il market making su eventi sintetici con valori veri noti.

La CfC a più uscite stima, dagli ultimi --lookback eventi irregolari, i valori in vigore
all'evento successivo; si confronta con i valori veri e con le stime "da manuale" della
finestra. --data events: drift e volatilità; --data book: anche tossicità e k.

    python -m experiments.check_event_model --events 60000 --epochs 15
    python -m experiments.check_event_model --data book
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from cfc.config import CfcConfig
from market.book_windows import OUTPUTS as BOOK_OUTPUTS
from market.book_windows import book_windows
from market.event_windows import event_windows
from market.synthetic_book import book_events
from market.synthetic_events import regime_events
from timeseries.evaluation import predict
from timeseries.model import MultiOutputForecaster
from timeseries.windows import Standardizer
from training.fit import fit_and_select
from training.metrics_log import MetricsLog
from training.multi_loss import MultiOutputLoss

OUTPUTS = ("drift", "volatility")
# Generatore, finestre e uscite per ogni tipo di dati sintetici.
DATASETS = {
    "events": (regime_events, event_windows, OUTPUTS),
    "book": (book_events, book_windows, BOOK_OUTPUTS),
}


def constant_baselines(truth: torch.Tensor, training_mean: float) -> dict[str, float]:
    """Previsioni costanti: quella onesta usa la media del TRAINING; l'oracolo la media
    del test stesso (informazione non disponibile a priori, solo come limite inferiore).
    La differenza fra i due è il quadrato dello scarto fra le due medie.
    """
    truth = truth.double()
    return {
        "constant_mse": ((truth - training_mean) ** 2).mean().item(),
        "oracle_constant_mse": truth.var(unbiased=False).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", choices=sorted(DATASETS), default="events")
    parser.add_argument("--events", type=int, default=60000)
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--backbone-units", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto: GPU se disponibile, come gli altri script.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    # Tre serie indipendenti (semi diversi): training, validazione, test.
    generate, windows, outputs = DATASETS[args.data]
    parts = {name: windows(generate(args.events if name == "training" else args.events // 3,
                                    seed=args.seed * 10 + offset), lookback=args.lookback)
             for offset, name in enumerate(("training", "validation", "test"))}
    returns = Standardizer.fit(parts["training"]["inputs"][:, :, 0].flatten())
    target_scalers = [Standardizer.fit(parts["training"]["targets"][:, k]) for k in range(len(outputs))]

    def tensors(part):
        inputs = part["inputs"].clone()
        inputs[:, :, 0] = returns.transform(inputs[:, :, 0])   # Δt (e lato, portata) restano grezzi
        targets = torch.stack([s.transform(part["targets"][:, k])
                               for k, s in enumerate(target_scalers)], dim=1)
        return inputs.float(), targets.float()

    input_size = parts["training"]["inputs"].shape[2]
    model = MultiOutputForecaster(CfcConfig(input_size, args.hidden_size, backbone_units=args.backbone_units),
                                  outputs, elapsed_channel=1).to(device)
    loss = MultiOutputLoss(outputs)
    directory = Path("outputs") / f"event_model_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}"
    directory.mkdir(parents=True, exist_ok=False)
    validation = tensors(parts["validation"])
    started = time.perf_counter()
    fit_and_select(model, torch.optim.Adam(model.parameters(), lr=0.005), loss,
                   tuple(tensor.to(device) for tensor in tensors(parts["training"])), validation,
                   dt=1.0, epochs=args.epochs,
                   batch_size=128, final_epoch=True, predict=predict,
                   metrics=MetricsLog(directory / "metrics.csv",
                                      validation_baseline=validation[1].double().pow(2).mean().item()),
                   save_best=lambda *_: None)

    seconds = time.perf_counter() - started
    test_inputs, _ = tensors(parts["test"])
    standardized = predict(model, test_inputs, dt=1.0).double()
    estimates = torch.stack([s.inverse(standardized[:, k]) for k, s in enumerate(target_scalers)], dim=1)
    truth, naive = parts["test"]["targets"], parts["test"]["naive"]

    results = {}
    print(f"Test: {len(truth)} finestre di {args.lookback} eventi")
    for k, name in enumerate(outputs):
        results[name] = {
            "cfc_mse": ((estimates[:, k] - truth[:, k]) ** 2).mean().item(),
            "naive_mse": ((naive[:, k] - truth[:, k]) ** 2).mean().item(),
            **constant_baselines(truth[:, k], target_scalers[k].mean),
        }
        print(f"  {name:10s} MSE contro il vero: CfC {results[name]['cfc_mse']:.4f} | "
              f"finestra {results[name]['naive_mse']:.4f} | costante (media training) "
              f"{results[name]['constant_mse']:.4f} | oracolo {results[name]['oracle_constant_mse']:.4f}")
    (directory / "results.json").write_text(json.dumps(
        {"arguments": vars(args), "device": str(device), "training_seconds": seconds,
         "results": results}, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Dispositivo: {device} | training {seconds:.1f} s")
    print(f"Risultati in {directory}")


if __name__ == "__main__":
    main()
