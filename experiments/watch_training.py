"""Grafico dal vivo delle loss di un training in corso (legge metrics.csv).

    python -m experiments.watch_training            # ultimo training, finestra aggiornata
    python -m experiments.watch_training --once     # salva solo il PNG e termina
"""

import argparse
import csv
from pathlib import Path

import matplotlib


def read_metrics(path: Path) -> dict[str, list[float]]:
    """Colonne del CSV come liste; train_loss vuota (epoca 0) diventa NaN."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        name: [float(row[name]) if row[name] != "" else float("nan") for row in rows]
        for name in ("epoch", "seconds", "train_loss", "validation_loss", "validation_baseline")
    }


def latest_metrics(root: Path = Path("outputs")) -> Path:
    candidates = sorted(root.glob("timeseries_*/metrics.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("Nessun outputs/timeseries_*/metrics.csv trovato.")
    return candidates[-1]


def draw(axis, metrics, name):
    axis.clear()
    epochs = metrics["epoch"]
    axis.plot(epochs, metrics["train_loss"], label="training")
    axis.plot(epochs, metrics["validation_loss"], label="validazione")
    axis.axhline(metrics["validation_baseline"][0], color="0.4", linestyle="--",
                 label="validazione: prevedere sempre la media")
    best = min(range(len(epochs)), key=lambda k: metrics["validation_loss"][k])
    axis.plot(epochs[best], metrics["validation_loss"][best], "o", color="tab:red",
              label=f"migliore validazione (epoca {int(epochs[best])})")
    per_epoch = metrics["seconds"][-1] / max(epochs[-1], 1)
    axis.set_title(f"{name} — epoca {int(epochs[-1])}, {per_epoch:.0f} s/epoca")
    axis.set_xlabel("Epoca")
    axis.set_ylabel("MSE (rendimenti standardizzati)")
    axis.grid(alpha=0.25)
    axis.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, help="Predefinito: il training più recente.")
    parser.add_argument("--interval", type=float, default=5.0, help="Secondi fra gli aggiornamenti.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = args.metrics or latest_metrics()
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)

    if args.once:
        draw(axis, read_metrics(path), path.parent.name)
        output = path.with_name("loss.png")
        figure.savefig(output, dpi=120)
        print(f"Grafico salvato in {output}")
        return

    print(f"Seguo {path} (chiudere la finestra per uscire)")
    while plt.fignum_exists(figure.number):
        draw(axis, read_metrics(path), path.parent.name)
        plt.pause(args.interval)


if __name__ == "__main__":
    main()
