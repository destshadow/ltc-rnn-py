"""Command-line validation of the time-series training script."""

import pytest

from experiments.train_timeseries import parse_arguments


def test_negative_epochs_are_rejected_before_reading_data(tmp_path, capsys):
    missing = tmp_path / "not_read.csv"  # never created: parsing must fail first

    with pytest.raises(SystemExit) as error:
        parse_arguments(["--csv", str(missing), "--epochs", "-1"])

    assert error.value.code == 2
    assert "--epochs" in capsys.readouterr().err


@pytest.mark.parametrize("epochs", ["0", "5"])
def test_zero_and_positive_epochs_are_accepted(epochs):
    args = parse_arguments(["--csv", "prices.csv", "--epochs", epochs])
    assert args.epochs == int(epochs)


@pytest.mark.parametrize("option,invalid", [
    ("--lookback", "0"), ("--hidden-size", "0"), ("--substeps", "-1"),
    ("--batch-size", "0"), ("--max-skipped", "-1"),
])
def test_invalid_sizes_are_rejected_before_reading_data(option, invalid, capsys):
    with pytest.raises(SystemExit) as error:
        parse_arguments(["--csv", "not_read.csv", option, invalid])

    assert error.value.code == 2
    assert option in capsys.readouterr().err


def test_smallest_valid_sizes_are_accepted():
    args = parse_arguments([
        "--csv", "prices.csv", "--lookback", "1", "--hidden-size", "1",
        "--substeps", "1", "--batch-size", "1", "--max-skipped", "0",
    ])
    assert (args.lookback, args.hidden_size, args.substeps, args.batch_size,
            args.max_skipped) == (1, 1, 1, 1, 0)
