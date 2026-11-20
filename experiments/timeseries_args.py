"""Argomenti della riga di comando di experiments.train_timeseries."""

import argparse
from datetime import datetime
from pathlib import Path

from timeseries.calendar import CALENDARS


def non_negative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("deve essere un intero non negativo.")
    return value


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("deve essere un intero positivo.")
    return value


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Previsione del rendimento successivo con la LTC, confrontata con i riferimenti."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV con intestazione.")
    source.add_argument("--arff", type=Path, help="File ARFF, es. Bitstamp.")
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--period-hours", type=float, default=24.0)
    parser.add_argument("--calendar", choices=CALENDARS, default="continuous")
    parser.add_argument(
        "--max-skipped", type=non_negative_int, default=0,
        help="Periodi attesi mancanti tollerati senza spezzare la serie.",
    )
    parser.add_argument("--lookback", type=positive_int, default=30)
    parser.add_argument("--hidden-size", type=positive_int, default=8)
    parser.add_argument("--substeps", type=positive_int, default=6)
    parser.add_argument("--dt", type=float, default=0.1)
    # 0 valuta il modello iniziale senza training; i negativi sono rifiutati.
    parser.add_argument("--epochs", type=non_negative_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    # Senza scelta dell'epoca sulla validazione: resta una verifica non usata per decidere.
    parser.add_argument("--final-epoch", action="store_true",
                        help="Usa il modello dell'ultima epoca invece del migliore in validazione.")
    # Divisione a date fisse (AAAA-MM-GG): sostituisce le frazioni, così varianti
    # del training si confrontano sulla stessa validazione.
    parser.add_argument("--train-start", type=datetime.fromisoformat)
    parser.add_argument("--validation-start", type=datetime.fromisoformat)
    parser.add_argument("--test-start", type=datetime.fromisoformat)
    args = parser.parse_args(argv)

    if (args.validation_start is None) != (args.test_start is None):
        parser.error("--validation-start e --test-start vanno indicate insieme.")
    if args.train_start is not None and args.validation_start is None:
        parser.error("--train-start richiede anche --validation-start e --test-start.")
    return args
