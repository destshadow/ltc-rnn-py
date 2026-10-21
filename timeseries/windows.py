from dataclasses import dataclass

import torch


def log_returns(prices: torch.Tensor) -> torch.Tensor:
    """Rendimenti logaritmici fra prezzi consecutivi: [n] -> [n - 1]."""
    if prices.ndim != 1 or len(prices) < 2:
        raise ValueError("prices deve avere forma [n] con almeno due valori.")

    if not torch.isfinite(prices).all().item() or (prices <= 0).any().item():
        raise ValueError("I prezzi devono essere finiti e positivi.")

    return torch.log(prices[1:] / prices[:-1])


def chronological_split(
    series: torch.Tensor,
    *,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Divide la serie in tre tratti contigui: training, validazione, test.

    Le finestre vanno create dentro ciascun tratto: così nessun ingresso o
    obiettivo di validazione e test proviene da un tratto precedente.
    """
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} deve essere compreso tra 0 e 1.")

    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validazione e test devono lasciare dati al training.")

    total = len(series)
    test_size = int(total * test_fraction)
    validation_size = int(total * validation_fraction)
    training_size = total - validation_size - test_size

    if min(training_size, validation_size, test_size) == 0:
        raise ValueError("Serie troppo corta per la divisione richiesta.")

    validation_end = training_size + validation_size

    return (
        series[:training_size],
        series[training_size:validation_end],
        series[validation_end:],
    )


def walk_forward_folds(
    total: int,
    *,
    initial: int,
    step: int,
) -> list[tuple[range, range]]:
    """Coppie (training, test) in avanti nel tempo, con training crescente."""
    for name, value in (("total", total), ("initial", initial), ("step", step)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} deve essere un intero positivo.")

    folds = []
    end = initial

    while end + step <= total:
        folds.append((range(0, end), range(end, end + step)))
        end += step

    if not folds:
        raise ValueError("Serie troppo corta per almeno una piega.")

    return folds


@dataclass(frozen=True)
class Standardizer:
    """Media e deviazione stimate solo sul training."""

    mean: float
    std: float

    @classmethod
    def fit(cls, training: torch.Tensor) -> "Standardizer":
        if training.ndim != 1 or len(training) < 2:
            raise ValueError("Servono almeno due valori di training.")

        std = training.std().item()
        if not std > 0:
            raise ValueError("La deviazione standard del training è nulla.")

        return cls(mean=training.mean().item(), std=std)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def make_windows(
    series: torch.Tensor,
    *,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finestre [n, lookback, 1] e obiettivo successivo [n].

    La finestra i contiene series[i:i + lookback]; l'obiettivo è
    series[i + lookback], sempre successivo all'ultimo ingresso.
    """
    if series.ndim != 1:
        raise ValueError("series deve avere forma [n].")

    if type(lookback) is not int or lookback <= 0:
        raise ValueError("lookback deve essere un intero positivo.")

    count = len(series) - lookback
    if count <= 0:
        raise ValueError("Serie più corta della finestra richiesta.")

    inputs = series.unfold(0, lookback, 1)[:count].unsqueeze(-1)
    targets = series[lookback:]

    return inputs.contiguous(), targets.clone()
