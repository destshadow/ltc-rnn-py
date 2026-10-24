"""Reading price files: incomplete rows versus missing prices."""

from datetime import datetime

import pytest

from timeseries.prices import iter_arff_prices, iter_csv_prices


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_csv_row_without_price_field_is_an_explicit_error(tmp_path):
    path = write(tmp_path, "prices.csv", "Date,Close\n2024-01-01,10\n2024-01-02\n")

    with pytest.raises(ValueError, match="Riga 3 incompleta: manca la colonna Close"):
        list(iter_csv_prices(path))


def test_csv_blank_line_is_skipped(tmp_path):
    # The csv module ignores blank lines: they are neither an error nor a price.
    path = write(tmp_path, "prices.csv", "Date,Close\n2024-01-01,10\n\n2024-01-03,12\n")

    assert len(list(iter_csv_prices(path))) == 2


def test_csv_missing_date_field_is_an_explicit_error(tmp_path):
    path = write(tmp_path, "prices.csv", "Close,Date\n10,2024-01-01\n11\n")

    with pytest.raises(ValueError, match="Riga 3 incompleta: manca la colonna Date"):
        list(iter_csv_prices(path))


@pytest.mark.parametrize("empty", ["", "?"])
def test_csv_empty_price_is_still_a_missing_price(tmp_path, empty):
    path = write(tmp_path, "prices.csv", f"Date,Close\n2024-01-01,10\n2024-01-02,{empty}\n")

    assert list(iter_csv_prices(path)) == [(datetime(2024, 1, 1), 10.0)]


def test_arff_incomplete_row_is_an_explicit_error(tmp_path):
    path = write(
        tmp_path, "prices.arff",
        "@ATTRIBUTE Timestamp STRING\n@ATTRIBUTE Open REAL\n@ATTRIBUTE Close REAL\n"
        "@DATA\n'2024-01-01 00:00:00',1.0,1.5\n'2024-01-01 00:01:00',2.0\n",
    )

    with pytest.raises(ValueError, match="Riga di dati 2 incompleta: 2 campi, almeno 3"):
        list(iter_arff_prices(path))
