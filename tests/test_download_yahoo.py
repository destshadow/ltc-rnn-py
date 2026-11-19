"""Yahoo downloader: pure helpers only, no network and no yfinance needed."""

from datetime import date

import pytest

from experiments.download_yahoo import completed_rows, to_csv
from timeseries.prices import iter_csv_prices


ROWS = [(date(2026, 11, 17), 7340.53), (date(2026, 11, 18), 7323.63), (date(2026, 11, 19), 7229.0)]


def test_today_is_dropped_as_possibly_partial():
    assert completed_rows(ROWS, today=date(2026, 11, 19)) == ROWS[:2]
    assert completed_rows(ROWS, today=date(2026, 11, 20)) == ROWS


@pytest.mark.parametrize("rows,today,match", [
    (ROWS[2:], date(2026, 11, 19), "Nessuna giornata completa"),  # only today's row
    ([ROWS[1], ROWS[0]], date(2026, 11, 20), "crescenti"),
])
def test_invalid_histories_are_rejected(rows, today, match):
    with pytest.raises(ValueError, match=match):
        completed_rows(rows, today=today)


def test_csv_is_readable_by_the_project_reader(tmp_path):
    path = tmp_path / "gspc.csv"
    path.write_bytes(to_csv(ROWS[:2]))

    prices = [price for _, price in iter_csv_prices(path)]
    assert prices == [7340.53, 7323.63]
