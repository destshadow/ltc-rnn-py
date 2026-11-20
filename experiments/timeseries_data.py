"""Preparazione dei dati comune agli script di training sulle serie temporali."""

from datetime import timedelta

import torch

from timeseries.calendar import split_segments
from timeseries.dataset import returns_by_segment, split_pieces, split_pieces_at
from timeseries.prices import iter_arff_prices, iter_csv_prices, resample_last
from timeseries.windows import Standardizer


def period_of(args):
    return timedelta(hours=args.period_hours)


def load_series(args):
    if args.csv is not None:
        samples = iter_csv_prices(
            args.csv, time_column=args.time_column or "Date", price_column=args.price_column,
        )
    else:
        samples = iter_arff_prices(
            args.arff, time_attribute=args.time_column or "Timestamp",
            price_attribute=args.price_column,
        )
    return resample_last(samples, period=period_of(args))


def load_parts(args):
    """Prezzi → tratti senza dati mancanti → training, validazione, test, e scaler.

    I dati mancanti spezzano la serie; le chiusure previste dal calendario no.
    Con le date fisse si usa split_pieces_at, altrimenti le frazioni.
    """
    starts, prices = load_series(args)
    segments = split_segments(starts, prices, period=period_of(args), calendar=args.calendar,
                              max_skipped=args.max_skipped)
    pieces = returns_by_segment(segments, period=period_of(args))
    if args.validation_start is not None:
        parts = split_pieces_at(pieces, train_start=args.train_start,
                                validation_start=args.validation_start, test_start=args.test_start)
    else:
        parts = split_pieces(pieces, validation_fraction=args.validation_fraction,
                             test_fraction=args.test_fraction)
    scaler = Standardizer.fit(torch.cat([piece.returns for piece in parts[0]]))
    return starts, prices, segments, parts, scaler
