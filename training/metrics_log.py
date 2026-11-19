import csv
import time
from pathlib import Path


FIELDS = ["epoch", "seconds", "train_loss", "validation_loss", "validation_baseline"]


class MetricsLog:
    """Scrive una riga per epoca in un CSV, subito su disco: si può seguire dal vivo."""

    def __init__(self, path: Path, *, validation_baseline: float):
        self.path = Path(path)
        self.validation_baseline = validation_baseline
        self.started = time.perf_counter()

        with self.path.open("x", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(FIELDS)

    def write(self, epoch: int, train_loss: float | None, validation_loss: float) -> None:
        row = [
            epoch,
            round(time.perf_counter() - self.started, 2),
            "" if train_loss is None else train_loss,  # epoca 0: nessun training
            validation_loss,
            self.validation_baseline,
        ]
        with self.path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)
